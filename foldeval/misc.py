from datetime import datetime
from typing import Optional
from pydrive.drive import GoogleDrive





def create_jobname_suffix() -> str:
  now = str(datetime.today())
  return now.replace('-', '').replace(' ', '').replace(':', '').replace('.', '')



def zip_and_download_folder(folder: str, 
                            drive: Optional[GoogleDrive] = None):
  from google.colab import files
  from colabtoolbox import gdrive
  import os
  zipped_folder = f'{folder}.zip'
  os.system(f'zip -q -r {zipped_folder} {folder}')
  if drive is None: 
    files.download(zipped_folder)
    return
  return gdrive.upload_to_google_drive(drive, zipped_folder)
