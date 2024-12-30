python3 QG_train.py \
    --n_epochs 4 \
    --eval_before_start \
    --local_rank 0 \
    --train_batch_size 2 \
    --model_name_or_path vinai/bartpho-syllable  \
    --output_dir /kaggle/working/vn-qa-gen/output/bartpho-syllable \
    --train_dataset_path /kaggle/working/vn-qa-gen/datasets/ViQuAD1.0/train_ViQuAD.json \
    --dev_dataset_path /kaggle/working/vn-qa-gen/datasets//ViQuAD1.0/dev_ViQuAD.json \
    --train_dataset_cache_path /kaggle/working/vn-qa-gen/datasets/QG/train_cache.pkl \
    --dev_dataset_cache_path /kaggle/working/vn-qa-gen/datasets/QG/dev_cache.pkl

#     --model_name_or_path /kaggle/working/vn-qa-gen/model_param  \
