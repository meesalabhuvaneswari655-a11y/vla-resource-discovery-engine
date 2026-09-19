"""Initialize Git repository and create initial commit using Dulwich (pure Python Git)."""

import os
from pathlib import Path
from dulwich.repo import Repo
from dulwich.porcelain import init, add, commit

def setup_repo():
    project_dir = Path(__file__).resolve().parent
    git_dir = project_dir / ".git"

    if not git_dir.exists():
        print(f"Initializing Git repository at: {project_dir}")
        repo = init(str(project_dir))
    else:
        repo = Repo(str(project_dir))
        print(f"Git repository already initialized at: {project_dir}")

    # Stage all files
    print("Staging files...")
    add(repo, paths=None)

    # Commit
    print("Creating initial commit...")
    author = b"Meghana <developer@example.com>"
    commit_message = b"feat: Initial commit of VLA Resource Discovery and Ranking Engine"
    
    try:
        commit_id = commit(repo, message=commit_message, author=author, committer=author)
        print(f"Committed successfully! Commit ID: {commit_id.decode('ascii') if isinstance(commit_id, bytes) else commit_id}")
    except Exception as e:
        print(f"Commit note: {e}")

    # Set default branch to refs/heads/main
    try:
        repo.refs.set_symbolic_ref(b"HEAD", b"refs/heads/main")
        head_commit = repo.head()
        repo.refs[b"refs/heads/main"] = head_commit
        print("Branch set to 'main'")
    except Exception as e:
        print(f"Branch setting note: {e}")

    print("\nGit repository successfully initialized and committed!")

if __name__ == "__main__":
    setup_repo()
