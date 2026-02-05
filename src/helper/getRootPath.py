from pathlib import Path
from config import PROJECT_NAME

def getRootPath() -> str:
    currentPath = str(Path.cwd())
    projectNameLength = len(PROJECT_NAME)
    return currentPath[0: currentPath.find(PROJECT_NAME) + projectNameLength]