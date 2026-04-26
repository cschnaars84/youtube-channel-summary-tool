from __future__ import annotations

import json
import os
import re
import secrets
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlparse

import requests
import streamlit as st
from openai import OpenAI
from youtube_transcript_api import YouTubeTranscriptApi
from yt_dlp import YoutubeDL

BASE_DIR = Path(__file__).resolve().parent
CHANNELS_FILE = BASE_DIR / "channels.json"
VIDEO_STATE_FILE = BASE_DIR / "video_state.json"
VIDEO_META_CACHE_FILE = BASE_DIR / "video_meta_cache.json"
APP_DB_FILE = BASE_DIR / "app.db"

DEFAULT_SUMMARY_MODEL = "gpt-4o-mini"


def inject_modern_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: #f3f4f6;
        }
        [data-testid="stHeader"] {
            background: transparent;
            height: 0rem;
        }
        [data-testid="stToolbar"] {
            right: 0.75rem;
            top: 0.35rem;
        }
        .block-container {
            padding-top: 0.6rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }
        [data-testid="stSidebar"] {
            border-right: 1px solid #e5e7eb;
            background: #f7f7f8;
        }
        .hero {
            border: 1px solid #e5e7eb;
            background: #f3f4f6;
            border-radius: 16px;
            padding: 1rem 1.2rem;
            margin-bottom: 1rem;
            box-shadow: 0 6px 20px rgba(15, 23, 42, 0.06);
        }
        .app-title {
            font-size: 2.1rem;
            font-weight: 800;
            margin-bottom: 0.2rem;
            letter-spacing: -0.5px;
            color: #111827;
        }
        .app-subtitle {
            color: #6b7280;
            margin-bottom: 0.3rem;
        }
        .video-card {
            background: #f8fafc;
            border: 1px solid #e5e7eb;
            border-radius: 16px;
            padding: 0.9rem 1rem;
            margin-bottom: 0.85rem;
            box-shadow: 0 6px 20px rgba(15, 23, 42, 0.06);
        }
        .status-read,
        .status-unread {
            display: inline-block;
            border-radius: 999px;
            padding: 0.2rem 0.6rem;
            font-size: 0.78rem;
            font-weight: 700;
            margin-bottom: 0.55rem;
        }
        .status-read {
            background: #ecfdf3;
            color: #166534;
            border: 1px solid #bbf7d0;
        }
        .status-unread {
            background: #fff7ed;
            color: #9a3412;
            border: 1px solid #fed7aa;
        }
        .video-title {
            font-size: 1.1rem;
            font-weight: 700;
            line-height: 1.35;
            margin-bottom: 0.35rem;
            color: #111827;
        }
        .video-title.read {
            color: #6b7280;
            text-decoration: line-through;
        }
        .video-meta {
            color: #4b5563;
            font-size: 0.9rem;
            margin-bottom: 0.2rem;
        }
        .video-date {
            color: #1d4ed8;
            font-size: 0.9rem;
            margin-bottom: 0.35rem;
            font-weight: 600;
        }
        div[data-testid="stButton"] > button {
            border-radius: 10px;
            border: 1px solid #d1d5db;
            background: #f3f4f6;
            color: #111827;
            font-weight: 600;
        }
        div[data-testid="stButton"] > button:hover {
            border-color: #9ca3af;
            background: #e5e7eb;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_db_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(APP_DB_FILE)
    connection.row_factory = sqlite3.Row
    return connection


def init_db() -> None:
    with get_db_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                google_sub TEXT UNIQUE NOT NULL,
                email TEXT NOT NULL,
                name TEXT,
                picture TEXT,
                created_at TEXT NOT NULL,
                last_login_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS channels (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                url TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(user_id, url),
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS video_states (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                video_id TEXT NOT NULL,
                title TEXT,
                url TEXT,
                channel_name TEXT,
                timestamp INTEGER,
                published_at TEXT,
                read INTEGER DEFAULT 0,
                insights_json TEXT,
                updated_at TEXT NOT NULL,
                UNIQUE(user_id, video_id),
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
            """
        )


def utc_now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def upsert_user(user_info: dict[str, Any]) -> int:
    google_sub = user_info.get("sub")
    email = user_info.get("email", "")
    name = user_info.get("name", "")
    picture = user_info.get("picture", "")
    now = utc_now_iso()
    with get_db_connection() as connection:
        connection.execute(
            """
            INSERT INTO users (google_sub, email, name, picture, created_at, last_login_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(google_sub) DO UPDATE SET
                email = excluded.email,
                name = excluded.name,
                picture = excluded.picture,
                last_login_at = excluded.last_login_at
            """,
            (google_sub, email, name, picture, now, now),
        )
        row = connection.execute("SELECT id FROM users WHERE google_sub = ?", (google_sub,)).fetchone()
    return int(row["id"])


def load_channels(user_id: int) -> list[dict[str, str]]:
    with get_db_connection() as connection:
        rows = connection.execute(
            "SELECT title, url FROM channels WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,),
        ).fetchall()
    return [{"title": row["title"], "url": row["url"]} for row in rows]


def insert_channel(user_id: int, title: str, url: str) -> None:
    with get_db_connection() as connection:
        connection.execute(
            """
            INSERT OR IGNORE INTO channels (user_id, title, url, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (user_id, title, url, utc_now_iso()),
        )


def delete_channel(user_id: int, url: str) -> None:
    with get_db_connection() as connection:
        connection.execute("DELETE FROM channels WHERE user_id = ? AND url = ?", (user_id, url))


def load_video_state(user_id: int) -> dict[str, Any]:
    with get_db_connection() as connection:
        rows = connection.execute(
            """
            SELECT video_id, title, url, channel_name, timestamp, published_at, read, insights_json
            FROM video_states
            WHERE user_id = ?
            """,
            (user_id,),
        ).fetchall()
    videos: dict[str, Any] = {}
    for row in rows:
        insights = []
        if row["insights_json"]:
            try:
                insights = json.loads(row["insights_json"])
            except json.JSONDecodeError:
                insights = []
        videos[row["video_id"]] = {
            "title": row["title"] or "",
            "url": row["url"] or "",
            "channel_name": row["channel_name"] or "",
            "timestamp": row["timestamp"],
            "published_at": row["published_at"] or "",
            "read": bool(row["read"]),
            "insights": insights,
        }
    return {"videos": videos}


def upsert_video_state(user_id: int, video_id: str, state: dict[str, Any]) -> None:
    with get_db_connection() as connection:
        connection.execute(
            """
            INSERT INTO video_states (
                user_id, video_id, title, url, channel_name, timestamp, published_at, read, insights_json, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id, video_id) DO UPDATE SET
                title = excluded.title,
                url = excluded.url,
                channel_name = excluded.channel_name,
                timestamp = excluded.timestamp,
                published_at = excluded.published_at,
                read = excluded.read,
                insights_json = excluded.insights_json,
                updated_at = excluded.updated_at
            """,
            (
                user_id,
                video_id,
                state.get("title", ""),
                state.get("url", ""),
                state.get("channel_name", ""),
                state.get("timestamp"),
                state.get("published_at", ""),
                1 if state.get("read") else 0,
                json.dumps(state.get("insights", []), ensure_ascii=False),
                utc_now_iso(),
            ),
        )


def get_openai_api_key() -> str:
    # Streamlit Cloud secrets first, local env fallback for development.
    try:
        secret_value = st.secrets.get("OPENAI_API_KEY", "")
    except Exception:
        secret_value = ""
    return secret_value or os.getenv("OPENAI_API_KEY", "")


def get_openai_model() -> str:
    try:
        model_from_secret = st.secrets.get("OPENAI_MODEL", "")
    except Exception:
        model_from_secret = ""
    return model_from_secret or os.getenv("OPENAI_MODEL", DEFAULT_SUMMARY_MODEL)


def get_google_oauth_config() -> dict[str, str]:
    try:
        return {
            "client_id": st.secrets.get("GOOGLE_CLIENT_ID", ""),
            "client_secret": st.secrets.get("GOOGLE_CLIENT_SECRET", ""),
            "redirect_uri": st.secrets.get("GOOGLE_REDIRECT_URI", ""),
        }
    except Exception:
        return {"client_id": "", "client_secret": "", "redirect_uri": ""}


def build_google_login_url() -> str:
    config = get_google_oauth_config()
    state = secrets.token_urlsafe(24)
    st.session_state.oauth_state = state
    params = {
        "client_id": config["client_id"],
        "redirect_uri": config["redirect_uri"],
        "response_type": "code",
        "scope": "openid email profile",
        "state": state,
        "include_granted_scopes": "true",
    }
    return f"https://accounts.google.com/o/oauth2/v2/auth?{urlencode(params)}"


def exchange_code_for_user_info(code: str) -> dict[str, Any]:
    config = get_google_oauth_config()
    token_response = requests.post(
        "https://oauth2.googleapis.com/token",
        data={
            "code": code,
            "client_id": config["client_id"],
            "client_secret": config["client_secret"],
            "redirect_uri": config["redirect_uri"],
            "grant_type": "authorization_code",
        },
        timeout=20,
    )
    token_response.raise_for_status()
    access_token = token_response.json().get("access_token")
    profile_response = requests.get(
        "https://openidconnect.googleapis.com/v1/userinfo",
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=20,
    )
    profile_response.raise_for_status()
    return profile_response.json()


def handle_google_callback() -> None:
    query_params = st.query_params
    if "code" not in query_params:
        return
    code = query_params.get("code")
    state = query_params.get("state", "")
    code_value = code if isinstance(code, str) else (code[0] if code else "")
    state_value = state if isinstance(state, str) else (state[0] if state else "")
    expected_state = st.session_state.get("oauth_state", "")
    # Streamlit can start a fresh session after OAuth redirect.
    # If we have an expected state, enforce it. If not, continue.
    if not code_value:
        st.error("Login fehlgeschlagen (ungueltiger OAuth-Status).")
        return
    if expected_state and state_value != expected_state:
        st.error("Login fehlgeschlagen (ungueltiger OAuth-Status).")
        return
    try:
        user_info = exchange_code_for_user_info(code_value)
        user_id = upsert_user(user_info)
    except Exception as error:
        st.error(f"Google Login fehlgeschlagen: {error}")
        return
    st.session_state.auth_user = user_info
    st.session_state.user_id = user_id
    st.query_params.clear()
    st.rerun()


def load_video_meta_cache() -> dict[str, Any]:
    if not VIDEO_META_CACHE_FILE.exists():
        return {"videos": {}}
    with VIDEO_META_CACHE_FILE.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if "videos" not in data or not isinstance(data["videos"], dict):
        return {"videos": {}}
    return data


def save_video_meta_cache(cache: dict[str, Any]) -> None:
    with VIDEO_META_CACHE_FILE.open("w", encoding="utf-8") as file:
        json.dump(cache, file, ensure_ascii=False, indent=2)


def get_ydl_opts(*, extract_flat: bool | str = True, playlistend: int | None = None) -> dict[str, Any]:
    opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        # Keep metadata in non-English localized form (prevents forced English titles).
        "extractor_args": {"youtube": {"lang": ["de"]}},
    }
    if extract_flat:
        opts["extract_flat"] = extract_flat
    if playlistend is not None:
        opts["playlistend"] = playlistend
    return opts


def normalize_channel_url(channel_url: str) -> str:
    raw = channel_url.strip()
    if not raw:
        return raw
    if not raw.startswith(("http://", "https://")):
        raw = f"https://{raw}"
    parsed = urlparse(raw)
    netloc = parsed.netloc.lower().replace("m.youtube.com", "youtube.com")
    path = parsed.path.rstrip("/")

    if not path:
        return f"https://{netloc}"
    if path.endswith("/videos"):
        return f"https://{netloc}{path}"
    return f"https://{netloc}{path}/videos"


def fallback_channel_title(channel_url: str) -> str:
    path = urlparse(channel_url).path.strip("/")
    if not path:
        return "YouTube Channel"
    if path.startswith("@"):
        return path[1:]
    return path.split("/")[-1]


def extract_video_id(raw_value: Any) -> str | None:
    if not raw_value:
        return None
    text = str(raw_value).strip()
    # Plain YouTube ID
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", text):
        return text
    # URL forms
    parsed = urlparse(text)
    if parsed.query:
        for part in parsed.query.split("&"):
            if part.startswith("v="):
                candidate = part[2:]
                if re.fullmatch(r"[A-Za-z0-9_-]{11}", candidate):
                    return candidate
    path_parts = [part for part in parsed.path.split("/") if part]
    if path_parts:
        candidate = path_parts[-1]
        if re.fullmatch(r"[A-Za-z0-9_-]{11}", candidate):
            return candidate
    return None


@st.cache_data(ttl=3600, show_spinner=False)
def resolve_channel_title(channel_url: str) -> str:
    ydl_opts = get_ydl_opts(extract_flat=True)
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(channel_url, download=False)
    return info.get("channel") or info.get("uploader") or info.get("title") or fallback_channel_title(channel_url)


@st.cache_data(ttl=900, show_spinner=False)
def fetch_latest_videos(channel_url: str, limit: int = 10) -> list[dict[str, Any]]:
    # Flat tab extraction is much faster and avoids heavy per-video format probing.
    ydl_opts = get_ydl_opts(extract_flat="in_playlist", playlistend=limit)
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(channel_url, download=False)

    entries = info.get("entries", []) or []
    videos = []
    for entry in entries:
        video_id = extract_video_id(entry.get("id")) or extract_video_id(entry.get("url"))
        if not video_id:
            continue
        thumbnail = entry.get("thumbnail") or f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg"
        videos.append(
            {
                "id": video_id,
                "title": entry.get("title", "Ohne Titel"),
                "url": f"https://www.youtube.com/watch?v={video_id}",
                "uploader": entry.get("uploader") or info.get("channel"),
                "upload_date": entry.get("upload_date", "Unbekannt"),
                "timestamp": entry.get("timestamp"),
                "duration": entry.get("duration"),
                "view_count": entry.get("view_count"),
                "thumbnail": thumbnail,
            }
        )
    return videos


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_video_details(video_id: str) -> dict[str, Any]:
    ydl_opts = get_ydl_opts(extract_flat=False)
    ydl_opts["noplaylist"] = True
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
    return {
        "title": info.get("title"),
        "upload_date": info.get("upload_date"),
        "timestamp": info.get("timestamp"),
        "duration": info.get("duration"),
        "view_count": info.get("view_count"),
        "thumbnail": info.get("thumbnail"),
    }


def enrich_videos_with_missing_metadata(videos: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cache = st.session_state.video_meta_cache.setdefault("videos", {})
    missing_ids: list[str] = []

    for video in videos:
        cached = cache.get(video["id"], {})
        for key in ("title", "upload_date", "timestamp", "duration", "view_count", "thumbnail"):
            if video.get(key) in (None, "", "Unbekannt") and cached.get(key) not in (None, "", "Unbekannt"):
                video[key] = cached[key]
        if not video.get("thumbnail"):
            video["thumbnail"] = f"https://i.ytimg.com/vi/{video['id']}/hqdefault.jpg"
        needs_details = any(
            video.get(field) in (None, "", "Unbekannt")
            for field in ("upload_date", "timestamp", "duration", "view_count")
        )
        if needs_details:
            missing_ids.append(video["id"])

    details_by_id: dict[str, dict[str, Any]] = {}
    if missing_ids:
        # Parallelize detail fetches to reduce waiting time.
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_map = {executor.submit(fetch_video_details, video_id): video_id for video_id in missing_ids}
            for future in as_completed(future_map):
                video_id = future_map[future]
                try:
                    details_by_id[video_id] = future.result()
                except Exception:
                    details_by_id[video_id] = {}

    enriched: list[dict[str, Any]] = []
    for video in videos:
        details = details_by_id.get(video["id"], {})
        merged = {**video}
        for key, value in details.items():
            if value not in (None, ""):
                merged[key] = value
        if not merged.get("thumbnail"):
            merged["thumbnail"] = f"https://i.ytimg.com/vi/{video['id']}/hqdefault.jpg"
        cache[video["id"]] = {
            "title": merged.get("title"),
            "upload_date": merged.get("upload_date"),
            "timestamp": merged.get("timestamp"),
            "duration": merged.get("duration"),
            "view_count": merged.get("view_count"),
            "thumbnail": merged.get("thumbnail"),
        }
        enriched.append(merged)
    save_video_meta_cache(st.session_state.video_meta_cache)
    return enriched


def fetch_transcript(video_id: str) -> str:
    # Compatibility with different youtube-transcript-api versions.
    def normalize_segments(raw_segments: Any) -> list[str]:
        texts: list[str] = []
        snippets = getattr(raw_segments, "snippets", raw_segments)
        for segment in snippets:
            if isinstance(segment, dict):
                text = segment.get("text", "").strip()
            else:
                text = getattr(segment, "text", "").strip()
            if text:
                texts.append(text)
        return texts

    try:
        api = YouTubeTranscriptApi()
        if hasattr(api, "fetch"):
            segments = api.fetch(video_id, languages=["de", "de-DE", "en"])
            texts = normalize_segments(segments)
            if texts:
                return " ".join(texts)
    except Exception:
        pass

    if hasattr(YouTubeTranscriptApi, "get_transcript"):
        try:
            segments = YouTubeTranscriptApi.get_transcript(video_id, languages=["de", "de-DE", "en"])
            texts = normalize_segments(segments)
            if texts:
                return " ".join(texts)
        except Exception:
            pass

    if hasattr(YouTubeTranscriptApi, "list_transcripts"):
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        try:
            transcript = transcript_list.find_transcript(["de", "de-DE", "en"])
        except Exception:
            transcript = transcript_list.find_generated_transcript(["de", "de-DE", "en"])
        segments = transcript.fetch()
        texts = normalize_segments(segments)
        if texts:
            return " ".join(texts)

    raise RuntimeError("Kein verfuegbares Transkript fuer dieses Video gefunden.")


def generate_insights_with_llm(transcript: str, video_title: str, api_key: str, model: str) -> list[str]:
    trimmed_transcript = transcript[:18000]
    client = OpenAI(api_key=api_key)
    completion = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": (
                    "Du bist ein praeziser Redakteur. Gib die wichtigsten Erkenntnisse in eigenen Worten wieder, "
                    "nicht als direkte Zitate aus dem Transkript."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Video-Titel: {video_title}\n\n"
                    "Fasse den Inhalt in 5-7 Erkenntnissen zusammen.\n"
                    "Wichtig:\n"
                    "- Jede Erkenntnis soll den Zusammenhang erklaeren.\n"
                    "- Schreibe klar und kurz, aber inhaltlich vollstaendig.\n"
                    "- Keine wortwoertlichen Zitate.\n"
                    "- Ausgabe nur als Aufzaehlung, eine Zeile pro Punkt mit '- '.\n\n"
                    f"Transkript:\n{trimmed_transcript}"
                ),
            },
        ],
    )
    content = completion.choices[0].message.content or ""
    insights = []
    for line in content.splitlines():
        cleaned = line.strip()
        if cleaned.startswith("- "):
            insights.append(cleaned[2:].strip())
        elif cleaned.startswith("• "):
            insights.append(cleaned[2:].strip())
    return [insight for insight in insights if insight]


def ensure_session_state() -> None:
    init_db()
    if "channels" not in st.session_state:
        st.session_state.channels = []
    if "videos_by_channel" not in st.session_state:
        st.session_state.videos_by_channel = {}
    if "video_state" not in st.session_state:
        st.session_state.video_state = {"videos": {}}
    if "video_meta_cache" not in st.session_state:
        st.session_state.video_meta_cache = load_video_meta_cache()
    if "auth_user" not in st.session_state:
        st.session_state.auth_user = None
    if "user_id" not in st.session_state:
        st.session_state.user_id = None
    if "loaded_user_id" not in st.session_state:
        st.session_state.loaded_user_id = None


def hydrate_user_state_if_needed() -> None:
    user_id = st.session_state.user_id
    if not user_id:
        return
    if st.session_state.loaded_user_id == user_id:
        return
    st.session_state.channels = load_channels(user_id)
    st.session_state.video_state = load_video_state(user_id)
    st.session_state.videos_by_channel = {}
    st.session_state.all_videos = []
    st.session_state.loaded_user_id = user_id


def add_channel(channel_url: str) -> None:
    channel_url = normalize_channel_url(channel_url)
    if not channel_url:
        st.warning("Bitte eine gueltige YouTube-Channel-URL eintragen.")
        return
    if any(channel["url"] == channel_url for channel in st.session_state.channels):
        st.info("Channel ist bereits gespeichert.")
        return

    try:
        title = resolve_channel_title(channel_url)
        insert_channel(st.session_state.user_id, title, channel_url)
        st.session_state.channels = load_channels(st.session_state.user_id)
        st.success(f"Channel hinzugefuegt: {title}")
    except Exception as error:
        st.error(f"Channel konnte nicht geladen werden: {error}")


def remove_channel(channel_url: str) -> None:
    delete_channel(st.session_state.user_id, channel_url)
    st.session_state.channels = load_channels(st.session_state.user_id)
    st.session_state.videos_by_channel.pop(channel_url, None)
    st.success("Channel entfernt.")


def format_upload_date(upload_date: str) -> str:
    if len(upload_date) == 8 and upload_date.isdigit():
        return f"{upload_date[6:8]}.{upload_date[4:6]}.{upload_date[0:4]}"
    return upload_date


def format_published_at(video: dict[str, Any]) -> str:
    timestamp = video.get("timestamp")
    if isinstance(timestamp, (int, float)):
        return datetime.fromtimestamp(timestamp).strftime("%d.%m.%Y %H:%M")
    return format_upload_date(video.get("upload_date", "Unbekannt"))


def format_duration(seconds: Any) -> str:
    if not isinstance(seconds, (int, float)) or seconds < 0:
        return "Unbekannt"
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours}:{minutes:02}:{secs:02}"
    return f"{minutes}:{secs:02}"


def format_view_count(view_count: Any) -> str:
    if not isinstance(view_count, int) or view_count < 0:
        return "Unbekannt"
    return f"{view_count:,}".replace(",", ".")


def get_video_state(video_id: str) -> dict[str, Any]:
    videos_state = st.session_state.video_state.get("videos", {})
    if video_id in videos_state:
        return videos_state[video_id]
    # Backward compatibility for older keys (full URL saved as key).
    legacy_key = f"https://www.youtube.com/watch?v={video_id}"
    return videos_state.get(legacy_key, {})


def update_video_state(video: dict[str, Any], *, insights: list[str] | None = None, read: bool | None = None) -> None:
    videos_state = st.session_state.video_state.setdefault("videos", {})
    current = videos_state.get(video["id"], {})
    if insights is not None:
        current["insights"] = insights
    if read is not None:
        current["read"] = read
    current["title"] = video["title"]
    current["url"] = video["url"]
    current["channel_name"] = video.get("channel_name", "")
    current["timestamp"] = video.get("timestamp")
    current["published_at"] = format_published_at(video)
    videos_state[video["id"]] = current
    upsert_video_state(st.session_state.user_id, video["id"], current)


def load_videos_for_all_channels(limit: int = 10) -> list[dict[str, Any]]:
    aggregated: list[dict[str, Any]] = []
    for channel in st.session_state.channels:
        videos = fetch_latest_videos(channel["url"], limit)
        videos = enrich_videos_with_missing_metadata(videos)
        st.session_state.videos_by_channel[channel["url"]] = videos
        for video in videos:
            enriched = dict(video)
            enriched["channel_name"] = channel["title"]
            aggregated.append(enriched)
    return sorted(aggregated, key=lambda video: (video.get("timestamp") or 0), reverse=True)


def render_video_list(videos: list[dict[str, Any]], openai_api_key: str, model_name: str) -> None:
    for video in videos:
        state = get_video_state(video["id"])
        is_read = bool(state.get("read"))
        read_label = "Gelesen" if is_read else "Ungelesen"
        status_class = "status-read" if is_read else "status-unread"
        status_text = "GELESEN" if is_read else "NEU"
        published_at = format_published_at(video)

        with st.container():
            st.markdown('<div class="video-card">', unsafe_allow_html=True)
            col_thumb, col_info = st.columns([1, 3])
            with col_thumb:
                if video.get("thumbnail"):
                    st.image(video["thumbnail"], width="stretch")
                else:
                    st.caption("Kein Thumbnail")
            with col_info:
                st.markdown(f'<span class="{status_class}">{status_text}</span>', unsafe_allow_html=True)
                title_class = "video-title read" if is_read else "video-title"
                st.markdown(f'<div class="{title_class}">{video["title"]}</div>', unsafe_allow_html=True)
                channel_label = video.get("channel_name") or video.get("uploader") or "Unbekannter Channel"
                st.markdown(f'<div class="video-date">🗓️ {published_at}</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="video-meta">Channel: {channel_label} | Status: {read_label} | '
                    f'Laenge: {format_duration(video.get("duration"))} | Views: {format_view_count(video.get("view_count"))}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(f"[Video oeffnen]({video['url']})")

            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Wichtigste Erkenntnisse erstellen", key=f"insights-{video['id']}"):
                    with st.spinner("Transkript wird geladen und analysiert..."):
                        try:
                            if not openai_api_key:
                                st.error("Kein OpenAI Key gesetzt. Bitte `OPENAI_API_KEY` in Streamlit Secrets konfigurieren.")
                                continue
                            transcript = fetch_transcript(video["id"])
                            insights = generate_insights_with_llm(
                                transcript=transcript,
                                video_title=video["title"],
                                api_key=openai_api_key,
                                model=model_name.strip() or DEFAULT_SUMMARY_MODEL,
                            )
                            update_video_state(video, insights=insights, read=True)
                        except Exception as error:
                            st.error(f"Erkenntnisse nicht moeglich: {error}")
            with col_b:
                toggle_label = "Als ungelesen markieren" if is_read else "Als gelesen markieren"
                if st.button(toggle_label, key=f"read-toggle-{video['id']}"):
                    update_video_state(video, read=not is_read)

            latest_state = get_video_state(video["id"])
            insights = latest_state.get("insights", [])
            if insights:
                with st.expander("Wichtigste Erkenntnisse", expanded=False):
                    for insight in insights:
                        st.markdown(f"- {insight}")
            st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(page_title="YouTube Channel Summaries", layout="wide")
    ensure_session_state()
    handle_google_callback()
    hydrate_user_state_if_needed()
    inject_modern_styles()

    oauth_config = get_google_oauth_config()
    if not st.session_state.auth_user:
        st.markdown(
            (
                '<div class="hero">'
                '<div class="app-title">YouTube Intelligence Dashboard</div>'
                '<div class="app-subtitle">Bitte zuerst mit Google anmelden, um deine persoenliche Channel- und Videoliste zu nutzen.</div>'
                "</div>"
            ),
            unsafe_allow_html=True,
        )
        if not (oauth_config["client_id"] and oauth_config["client_secret"] and oauth_config["redirect_uri"]):
            st.error(
                "Google OAuth ist noch nicht konfiguriert. Setze in Streamlit Secrets: "
                "`GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, `GOOGLE_REDIRECT_URI`."
            )
            return
        st.link_button("Mit Google anmelden", build_google_login_url(), use_container_width=True)
        return

    total_channels = len(st.session_state.channels)
    read_count = sum(1 for video in st.session_state.video_state.get("videos", {}).values() if video.get("read"))
    st.markdown(
        (
            '<div class="hero">'
            '<div class="app-title">YouTube Intelligence Dashboard</div>'
            '<div class="app-subtitle">Behalte neue Videos, Lesestatus und wichtigste Erkenntnisse in einem modernen Workflow im Blick.</div>'
            f'<div class="video-meta">Kanaele: {total_channels} | Bereits gelesen: {read_count}</div>'
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    with st.sidebar:
        user = st.session_state.auth_user or {}
        st.caption(f"Angemeldet als: {user.get('email', 'Unbekannt')}")
        if st.button("Abmelden"):
            for key in ["auth_user", "user_id", "loaded_user_id", "channels", "video_state", "videos_by_channel", "all_videos"]:
                st.session_state.pop(key, None)
            st.rerun()
        st.divider()
        st.header("Kanaele")
        new_channel_url = st.text_input("Neue Channel-URL")
        if st.button("Channel speichern"):
            add_channel(new_channel_url)

        st.divider()
        st.subheader("Zusammenfassung")
        openai_api_key = get_openai_api_key()
        model_name = get_openai_model()
        if not openai_api_key:
            st.warning("Kein OpenAI API Key gefunden. Bitte in Streamlit Secrets `OPENAI_API_KEY` setzen.")

        if st.session_state.channels:
            removable = st.selectbox(
                "Channel entfernen",
                st.session_state.channels,
                format_func=lambda channel: channel["title"],
                key="remove_select",
            )
            if st.button("Ausgewaehlten Channel loeschen"):
                remove_channel(removable["url"])
        else:
            st.caption("Noch keine Kanaele gespeichert.")

    if not st.session_state.channels:
        st.info("Lege zuerst mindestens einen YouTube-Channel in der Sidebar an.")
        return

    selected_channel = st.selectbox(
        "Waehle einen Channel",
        st.session_state.channels,
        format_func=lambda channel: channel["title"],
    )

    col_view_a, col_view_b = st.columns(2)
    load_single = col_view_a.button("Videos vom ausgewaehlten Channel laden")
    load_all = col_view_b.button("Videos aller Kanaele (chronologisch) laden")

    if load_single:
        with st.spinner("Videos werden geladen..."):
            try:
                videos = fetch_latest_videos(selected_channel["url"], 10)
                videos = enrich_videos_with_missing_metadata(videos)
                videos = [{**video, "channel_name": selected_channel["title"]} for video in videos]
                st.session_state.videos_by_channel[selected_channel["url"]] = videos
                st.session_state.active_view = "single"
                if not videos:
                    st.warning("Keine Videos gefunden. Bitte pruefe die Channel-URL.")
            except Exception as error:
                st.error(f"Videos konnten nicht geladen werden: {error}")
                return

    if load_all:
        with st.spinner("Lade aktuelle Videos aller Kanaele..."):
            try:
                st.session_state.all_videos = load_videos_for_all_channels(limit=10)
                st.session_state.active_view = "all"
            except Exception as error:
                st.error(f"Videos konnten nicht geladen werden: {error}")
                return

    active_view = st.session_state.get("active_view", "single")
    if active_view == "all":
        all_videos = st.session_state.get("all_videos", [])
        if not all_videos:
            st.caption("Noch keine kanaluebergreifenden Videos geladen.")
            return
        st.subheader("Alle aktuellen Videos (chronologisch)")
        render_video_list(all_videos, openai_api_key, model_name)
    else:
        videos = st.session_state.videos_by_channel.get(selected_channel["url"], [])
        if not videos:
            st.caption("Noch keine Videos geladen. Klicke auf den Lade-Button.")
            return
        st.subheader(f"Letzte Videos: {selected_channel['title']}")
        render_video_list(videos, openai_api_key, model_name)


if __name__ == "__main__":
    main()
