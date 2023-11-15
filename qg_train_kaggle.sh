python3 QG_train.py \
    --n_epochs 4 \
    --local_rank 0 \
    --model_name_or_path vinai/bartpho-syllable \
    --output_dir /kaggle/working/vn-qa-gen/output/bartpho-syllable \
    --train_dataset_path /kaggle/working/vn-qa-gen/Datasets/ViQuAD1.0/train_ViQuAD.json \
    --dev_dataset_path /kaggle/working/vn-qa-gen/Datasets//ViQuAD1.0/dev_ViQuAD.json \
    --train_dataset_cache_path /kaggle/working/vn-qa-gen/Datasets/QG/train_cache.pkl \
    --dev_dataset_cache_path /kaggle/working/vn-qa-gen/Datasets/QG/dev_cache.pkl

