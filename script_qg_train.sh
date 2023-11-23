python3 QG_train.py \
    --n_epochs 4 \
    --eval_before_start \
    --model_name_or_path vinai/bartpho-syllable \
    --output_dir /Users/phuongnguyen/study/vn-qa-gen/output/bartpho-syllable \
    --train_dataset_path /Users/phuongnguyen/study/vn-qa-gen/datasets/ViQuAD1.0/train_ViQuAD.json \
    --dev_dataset_path /Users/phuongnguyen/study/vn-qa-gen/datasets//ViQuAD1.0/dev_ViQuAD.json \
    --train_dataset_cache_path /Users/phuongnguyen/study/vn-qa-gen/datasets/QG/train_cache_debug.pkl \
    --dev_dataset_cache_path /Users/phuongnguyen/study/vn-qa-gen/datasets/QG/dev_cache_debug.pkl \
    --debug
