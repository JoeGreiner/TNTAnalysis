import git
from PyQt5.QtCore import QSettings

def check_if_git_uptodate():
    settings = QSettings("JoeGreiner", "TNT_Analysis_GUI")
    repo_path = settings.value("git_repo_path")
    repo = git.Repo(repo_path)

    local_branch = 'main'
    remote_branch = f'origin/{local_branch}'
    repo.remotes.origin.fetch()

    local_commit = repo.commit(local_branch)
    remote_commit = repo.commit(remote_branch)

    if local_commit == remote_commit:
        print("The local branch is up to date with the remote branch.")
        return True
    else:
        behind_commits = list(repo.iter_commits(f'{local_branch}..{remote_branch}'))
        print(f"The local branch is behind the remote branch by {len(behind_commits)} commits.")
        return False

