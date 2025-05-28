import os
import shutil

def copyInstanceJsonFile(
    originFolder: str,
    targetFolder: str
):
    """ Copy all files from the originFolder to the targetFolder if the file is a `.json` file. """
    if not os.path.exists(targetFolder):
        os.makedirs(targetFolder)
    
    for filename in os.listdir(originFolder):
        if filename.split(".")[-1] == "json":
            originFile = os.path.join(originFolder, filename)
            targetFile = os.path.join(targetFolder, filename)
            
            # Move each file
            shutil.copy(originFile, targetFile)

def copyInstanceTxtFile(
    originFolder: str,
    targetFolder: str
):
    """
    Copy all files from the originFolder to the targetFolder, if the file is a `.txt` file.
     """
    if not os.path.exists(targetFolder):
        os.makedirs(targetFolder)
    
    for filename in os.listdir(originFolder):
        if filename.split(".")[-1] == "txt":
            print("Add file: " + filename)
            originFile = os.path.join(originFolder, filename)
            targetFile = os.path.join(targetFolder, filename)
            shutil.copy(originFile, targetFile)