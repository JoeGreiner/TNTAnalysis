import logging
import git
from PyQt6.QtCore import QSettings

def check_if_git_uptodate():
    settings = QSettings("JoeGreiner", "TNT_Analysis_GUI")
    repo_path = settings.value("git_repo_path")
    local_branch = 'main'
    remote_branch = f'origin/{local_branch}'
    try:
        if not repo_path:
            raise ValueError('No Git repository path configured.')
        repo = git.Repo(repo_path)
        repo.remote('origin').fetch(kill_after_timeout=5)
        ahead, behind = map(int, repo.git.rev_list(
            '--left-right', '--count', f'{local_branch}...{remote_branch}').split())
    except (git.GitError, OSError, ValueError) as error:
        logging.warning('Could not check Git updates: %s', error)
        return None
    logging.info('Git status: %s is %d commit(s) ahead and %d behind %s.',
                 local_branch, ahead, behind, remote_branch)
    return behind == 0

