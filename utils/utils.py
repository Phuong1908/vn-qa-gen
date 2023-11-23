import os
import sys
import pickle
import json
import shutil
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

DRIVE_OUTPUT_PATH = '/Master_Thesis/Vietnamese/output'

def gg_auth():
  gauth = GoogleAuth()
  if gauth.credentials is None:
    # Authenticate if they're not there
    gauth.LocalWebserverAuth() 
  elif gauth.access_token_expired:
    # Refresh them if expired
    gauth.Refresh()
  else:
    # Initialize the saved creds
    gauth.Authorize()
  return gauth
  

def save_result_to_drive(epoch, prefix, path, score, checkpoint_file_path):
  filename = f'{prefix}.epoch_{epoch}'
  result_folder = f'{path}/{filename}'
  os.mkdir(result_folder)
  
  # save score into json file
  with open(f'{result_folder}/training_result.json', "w") as outfile: 
    json.dump(score, outfile)
    
  # move weight file to result folder
  move_file(checkpoint_file_path, result_folder)
     
  # compress folder
  shutil.make_archive(result_folder, 'zip', result_folder)

  #upload into drive
  gauth = gg_auth()
  drive = GoogleDrive(gauth)
  
  file_path = f'{path}/{filename}.zip'
  drive_file_path = f'{DRIVE_OUTPUT_PATH}/{filename}.zip'

  file_drive = drive.CreateFile({"title": drive_file_path}) 
  file_drive.SetContentFile(file_path)
  file_drive.Upload()
  print(f"Uploaded file with ID {file_drive.get('id')}")


def move_file(src_file, dst_dir):
    # check if source file exists
    if not os.path.isfile(src_file):
        print(f"Source file {src_file} does not exist.")
        return

    # check if destination directory exists
    if not os.path.isdir(dst_dir):
        print(f"Destination directory {dst_dir} does not exist.")
        return

    # get the base name of the file
    base_name = os.path.basename(src_file)

    # construct destination file path
    dst_file = os.path.join(dst_dir, base_name)

    # move the file
    shutil.move(src_file, dst_file)
    print(f"File {src_file} moved to {dst_file}")
    

def normalize_text(text):
    """
    Replace some special characters in text.
    """
    # NOTICE: don't change the text length.
    # Otherwise, the answer position is changed.
    text = text.replace("''", '" ').replace("``", '" ')
    return text


def pickle_dump_large_file(obj, filepath):
  """
  This is a defensive way to write pickle.write,
  allowing for very large files on all platforms
  """
  max_bytes = 2**31 - 1
  bytes_out = pickle.dumps(obj)
  n_bytes = sys.getsizeof(bytes_out)
  with open(filepath, 'wb') as f_out:
    for idx in range(0, n_bytes, max_bytes):
      f_out.write(bytes_out[idx:idx + max_bytes])
      
      
def pickle_load_large_file(filepath):
  """
  This is a defensive way to write pickle.load,
  allowing for very large files on all platforms
  """
  max_bytes = 2**31 - 1
  input_size = os.path.getsize(filepath)
  bytes_in = bytearray(0)
  with open(filepath, 'rb') as f_in:
    for _ in range(0, input_size, max_bytes):
      bytes_in += f_in.read(max_bytes)
  obj = pickle.loads(bytes_in)
  return obj


def save(filepath, obj, message=None):
  if message is not None:
    print("Saving {}...".format(message))
  pickle_dump_large_file(obj, filepath)
    
    
def load(filepath, message=None):
  if message is not None:
    print("Saving {}...".format(message))
  result = pickle_load_large_file(filepath)
  return result

if __name__ == "__main__":
  path = '/Users/phuongnguyen/study/vn-qa-gen/output/bartpho-syllable'
  score = {
    "nll": 111,
    "ppl": 111,
  }
  def save_checkpoint():
    print('checkpoint saved!')
    return
  
  save_result_to_drive(epoch=1, path=path, prefix='epx', score=score, save_checkpoint=save_checkpoint)
