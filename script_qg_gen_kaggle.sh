output_path="/Users/phuongnguyen/study/vn-qa-gen/datasets/processed/SQuAD2.0/"
data_file_prefix="train"
st_idx=0
ed_idx=1000
PYTHONIOENCODING=utf-8 python3 QG_gen.py  \
    --model_type vinai/bartpho-syllable \
    --model_name_or_path /Users/phuongnguyen/study/vn-qa-gen/output/bartpho-syllable \
    --filename "$output_path${data_file_prefix}.sentences.augmented.${st_idx}_${ed_idx}.pkl" \
    --filecache "$output_path${data_file_prefix}.sentences.augmented.${st_idx}_${ed_idx}.cache.qg.pth" \
    --data_type augmented_sents \
    --output_file "$output_path${data_file_prefix}.qa.${st_idx}_${ed_idx}.qg.generated.json"