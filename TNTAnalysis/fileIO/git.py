import logging
import git
from PyQt5.QtCore import QSettings

def check_if_git_uptodate():
    settings = QSettings("JoeGreiner", "TNT_Analysis_GUI")
    repo_path = settings.value("git_repo_path")
    repo = git.Repo(repo_path)

    local_branch = 'main'
    remote_branch = f'origin/{local_branch}'
    repo.remotes.origin.fetch()

    ahead, behind = map(int, repo.git.rev_list(
        '--left-right', '--count', f'{local_branch}...{remote_branch}').split())
    logging.info('Git status: %s is %d commit(s) ahead and %d behind %s.',
                 local_branch, ahead, behind, remote_branch)
    return behind == 0

