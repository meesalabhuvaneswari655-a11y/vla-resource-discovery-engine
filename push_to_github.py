"""Push local Git repository to GitHub using Dulwich (no git.exe required)."""

import argparse
import sys
from pathlib import Path
from dulwich.repo import Repo
from dulwich.porcelain import push

def push_repo(remote_url: str, token: str = None):
    project_dir = Path(__file__).resolve().parent
    repo = Repo(str(project_dir))

    # If a personal access token is provided, embed it into the HTTPS URL
    if token and "https://" in remote_url and "@" not in remote_url:
        target_url = remote_url.replace("https://", f"https://{token}@")
    else:
        target_url = remote_url

    print(f"Pushing project from: {project_dir}")
    print(f"Target GitHub URL: {remote_url}")

    try:
        push(repo, target_url, b"refs/heads/main:refs/heads/main")
        print("\nSuccessfully pushed all code and commit history to GitHub!")
    except Exception as e:
        print(f"\nPush encountered an issue: {e}")
        print("Tip: If GitHub requires authentication, supply your Personal Access Token:")
        print("py push_to_github.py --url <YOUR_GITHUB_URL> --token <YOUR_GITHUB_TOKEN>")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Push repository to GitHub.")
    parser.add_argument("--url", "-u", required=True, help="GitHub repository URL (https://github.com/...)")
    parser.add_argument("--token", "-t", required=False, default=None, help="GitHub Personal Access Token (classic or fine-grained)")
    args = parser.parse_args()

    push_repo(args.url, args.token)
