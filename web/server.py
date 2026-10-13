"""Kleiner lokaler Webserver für die Erklär-Website.

Start:  python web/server.py      ->  öffnet http://127.0.0.1:8000

- liefert die Website (Ordner web/) und die Dateien aus output/
- /api/run?name=<skript> startet eines der festen Skripte und schickt die Ausgabe live zurück
Nur Python-Standardbibliothek, lauscht nur auf 127.0.0.1 (nicht im Netzwerk erreichbar).
"""

import http.server
import os
import subprocess
import sys
import threading
import urllib.parse
import webbrowser

WEB = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(WEB)
PORT = 8000

# Nur diese Befehle dürfen gestartet werden (keine freien Eingaben vom Browser).
SCRIPTS = {
    "part2": ([sys.executable, "-u", "part2_train.py"], "python"),
    "part3": ([sys.executable, "-u", "part3_tensorflow.py"], "python"),
    "build": (["cmd", "/c", os.path.join(ROOT, "c", "build.bat")] if os.name == "nt" else ["sh", "-c", "cc -std=c99 -O2 -o part4_txt.exe part4_txt.c nn.c -lm && cc -std=c99 -O2 -I. -I../output -o part5_header.exe part5_header.c nn.c -lm && echo Fertig"], "c"),
    "part4": ([os.path.join(ROOT, "c", "part4_txt.exe")], "c"),
    "part5": ([os.path.join(ROOT, "c", "part5_header.exe")], "c"),
}


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB, **kwargs)

    def log_message(self, fmt, *args):  # Konsole ruhig halten
        pass

    def translate_path(self, path):
        # /output/... kommt aus dem Projektordner, alles andere aus web/
        clean = urllib.parse.urlparse(path).path
        if clean.startswith("/output/"):
            name = os.path.basename(urllib.parse.unquote(clean))
            return os.path.join(ROOT, "output", name)
        return super().translate_path(path)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        if url.path == "/api/run":
            name = urllib.parse.parse_qs(url.query).get("name", [""])[0]
            return self.run_script(name)
        return super().do_GET()

    def run_script(self, name):
        if name not in SCRIPTS:
            self.send_error(404, "Unbekanntes Skript")
            return
        cmd, folder = SCRIPTS[name]
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        env = dict(os.environ, PYTHONIOENCODING="utf-8", TF_CPP_MIN_LOG_LEVEL="3")
        try:
            proc = subprocess.Popen(cmd, cwd=os.path.join(ROOT, folder), stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, env=env)
        except OSError as e:
            self.wfile.write(f"Fehler beim Starten: {e}\n(Für Teil 4/5 zuerst 'C bauen' ausführen.)\n".encode())
            self.wfile.write(b"\n[EXIT 1]\n")
            return
        try:
            for line in iter(proc.stdout.readline, b""):
                text = line.decode("utf-8", errors="replace")
                if "oneDNN" in text or "absl::" in text or "GPU support" in text:
                    continue  # TensorFlow-Hinweise ausblenden
                self.wfile.write(text.encode("utf-8"))
                self.wfile.flush()
            proc.wait()
            self.wfile.write(f"\n[EXIT {proc.returncode}]\n".encode())
        except (BrokenPipeError, ConnectionResetError):
            proc.kill()  # Browser hat abgebrochen


def main():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    url = f"http://127.0.0.1:{PORT}/"
    print(f"Website läuft auf {url}  (Beenden mit Strg+C)")
    if "--no-browser" not in sys.argv:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
