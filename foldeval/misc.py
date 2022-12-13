from datetime import datetime





def create_jobname_suffix() -> str:
  now = str(datetime.today())
  return now.replace('-', '').replace(' ', '').replace(':', '').replace('.', '')
