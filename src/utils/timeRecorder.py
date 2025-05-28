from datetime import datetime

def getCurrentTimeForFileName(format: str="%Y%m%d_%H%M")-> str:
    now = datetime.now()
    return datetime.strftime(now, format)