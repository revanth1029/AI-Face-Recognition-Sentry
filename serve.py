import http.server
import socketserver
import os
import sys
import subprocess

PORT = 8000
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)
    
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        super().end_headers()
        
    def log_message(self, format, *args):
        # Print clean logs
        print(f"[HTTP] {self.address_string()} - {format % args}")

class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True

def open_chrome(url):
    chrome_paths = [
        os.path.expandvars(r"%ProgramFiles%\Google\Chrome\Application\chrome.exe"),
        os.path.expandvars(r"%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"),
        os.path.expandvars(r"%LocalAppData%\Google\Chrome\Application\chrome.exe")
    ]
    for p in chrome_paths:
        if os.path.exists(p):
            try:
                subprocess.Popen([p, url])
                print(f"Launched Chrome at {p}")
                return
            except Exception as e:
                print(f"Error launching Chrome: {e}")
    # Fallback
    import webbrowser
    webbrowser.open(url)

if __name__ == '__main__':
    url = f"http://localhost:{PORT}/"
    print("=" * 60)
    print(f"  AI Face Recognition Web Server Running at: {url}")
    print("  Serving directory:", DIRECTORY)
    print("  Press Ctrl+C to stop.")
    print("=" * 60)
    
    open_chrome(url)
    
    server = ThreadedHTTPServer(('localhost', PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        server.server_close()
