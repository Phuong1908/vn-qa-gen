import codecs 
import json
from tqdm import tqdm
from underthesea import sent_tokenize
from utils.utils import save, load
from config import *

def normalize_text(text):
    """
    Replace some special characters in text.
    """
    # NOTICE: don't change the text length.
    # Otherwise, the answer position is changed.
    text = text.replace("''", '" ').replace("``", '" ').replace("\t", " ")
    return text
  
def viquad2paragraph(input_path, output_path, paragraphs_path,
                    max_length=100, min_length=5, max_plength=400, min_plength=5):
  outfile = open(output_path, 'w+', encoding='utf8')
  outfile_p = open(paragraphs_path, 'w+', encoding='utf8')
  with codecs.open(input_path, "r", encoding='utf8') as infile:
    source = json.load(infile)
    pid = 0
    sid = 0
    for article in tqdm(source["data"]):
      for para in article["paragraphs"]:
        context = para["context"]
        p = context
        len_p = len(p.split())
        if len_p >= max_plength or len_p <= min_plength:
            continue
        p = normalize_text(p)
        outfile_p.write(str(pid) + "\t" + p.rstrip().replace("\n", "\\n") + "\n") 
        sentences = sent_tokenize(context)
        for s in sentences:
          len_s = len(s.split())
          if len_s >= max_length or len_s <= min_length:
              continue
          s = normalize_text(s)
          outfile.write(str(pid) + "\t" + str(sid) + "\t" + s.rstrip().replace("\n", "\\n") + "\n")
          sid += 1
        pid += 1
  infile.close()
  outfile.close()
  outfile_p.close()
  
  
def vinewsqa2paragraph(input_path, output_path, paragraphs_path,
                    max_length=100, min_length=5, max_plength=400, min_plength=5):
  outfile = open(output_path, 'w+', encoding='utf8')
  outfile_p = open(paragraphs_path, 'w+', encoding='utf8')
  with codecs.open(input_path, "r", encoding='utf8') as infile:
    source = json.load(infile)
    pid = 0
    sid = 0
    for article in tqdm(source["data"]):
      for para in article["paragraphs"]:
        context = para["context"]
        p = context
        len_p = len(p.split())
        if len_p >= max_plength or len_p <= min_plength:
            continue
        p = normalize_text(p)
        outfile_p.write(str(pid) + "\t" + p.rstrip().replace("\n", "\\n") + "\n")
        sentences = sent_tokenize(context)
        for s in sentences:
          len_s = len(s.split())
          if len_s >= max_length or len_s <= min_length:
              continue
          s = normalize_text(s)
          outfile.write(str(pid) + "\t" + str(sid) + "\t" + s.rstrip().replace("\n", "\\n") + "\n")
          sid += 1
        pid += 1
  infile.close()
  outfile.close()
  outfile_p.close()
  
  
def file2sentences(input_path, data_type, output_path, paragraphs_path,
                   max_length=100, min_length=5, max_plength=400, min_plength=5):
  if data_type.lower() == "vinewsqa":
    vinewsqa2paragraph(input_path, output_path, paragraphs_path, max_length, min_length, max_plength, min_plength)
  elif data_type.lower() == "viquad":
    viquad2paragraph(input_path, output_path, paragraphs_path, max_length, min_length, max_plength, min_plength)
  else:
    print("The data_type must be vinewsqa or viquad...")
    
    
def main(args):
  # excute tasks
  if args.debug:
    args.da_start_index = 0
    args.da_end_index = 10

  if args.da_task == "file2sentences":
    file2sentences(
      args.da_input_file,
      args.da_input_type,
      args.da_sentences_file,
      args.da_paragraphs_file,
      max_plength=args.para_limit,
      max_length=args.sent_limit)


if __name__ == '__main__':
  main(parser.parse_args())