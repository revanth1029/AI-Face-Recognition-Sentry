"""
AI Sentry - Automated Git Sync
Pushes updates to https://github.com/revanth1029/AI-Face-Recognition-Sentry
"""
import subprocess
import os
import sys
import datetime

GIT_PATH = r"C:\Users\srian\AppData\Local\GitHubDesktop\app-3.6.10\resources\app\git\cmd\git.exe"
if not os.path.exists(GIT_PATH):
    GIT_PATH = "git"

def sync_now(commit_msg=None):
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    if not commit_msg:
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        commit_msg = f"Auto-sync updates: {now_str}"
    
    subprocess.run([GIT_PATH, "add", "-A"])
    # Check if there are changes
    status = subprocess.run([GIT_PATH, "status", "--porcelain"], capture_output=True, text=True)
    if not status.stdout.strip():
        print("[Git Sync] Working tree is clean. Nothing new to commit.")
        return True

    res_commit = subprocess.run([GIT_PATH, "commit", "-m", commit_msg], capture_output=True, text=True)
    print(res_commit.stdout)
    
    print("[Git Sync] Pushing to origin main...")
    res_push = subprocess.run([GIT_PATH, "push", "origin", "main"], capture_output=True, text=True)
    print(res_push.stdout)
    if res_push.stderr:
        print(res_push.stderr)
    return res_push.returncode == 0

if __name__ == "__main__":
    msg = sys.argv[1] if len(sys.argv) > 1 else None
    sync_now(msg)
