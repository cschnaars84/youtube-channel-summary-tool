# YouTube Channel Summary Tool

Lokales Tool, mit dem du:
- wichtige YouTube-Channels speichern kannst,
- pro Channel die letzten 10 Videos laden kannst,
- ein Video auswaehlst und eine Zusammenfassung mit wichtigsten Erkenntnissen bekommst.

## Setup

```bash
cd "/Users/christianschnaars/Documents/youtube-channel-summary-tool"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## App starten

```bash
streamlit run app.py
```

Dann die angezeigte lokale URL im Browser oeffnen.

## Hinweise

- Gib am besten eine echte Channel-URL an, z. B.:
  - `https://www.youtube.com/@kurzgesagt`
  - `https://www.youtube.com/c/Fireship`
- Einige Videos haben keine verfuegbaren Untertitel/Transkripte. In dem Fall kann keine Zusammenfassung erstellt werden.
- Die Zusammenfassung ist eine automatische Extraktion aus dem Transkript.
