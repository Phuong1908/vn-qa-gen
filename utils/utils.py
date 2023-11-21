import os
import sys
import pickle

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