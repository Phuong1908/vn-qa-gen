python3 QG_train.py \
    --eval_before_start \
    --n_epochs 4 \
    --model_name_or_path bartpho-syllable \
    --output_dir /Users/phuongnguyen/study/vn-qa-gen/output/QG/gpt2_question_generation \
    --train_dataset_path /Users/phuongnguyen/study/vn-qa-gen/Datasets/ViQuAD1.0/train_ViQuAD.json \
    --dev_dataset_path /Users/phuongnguyen/study/vn-qa-gen/Datasets//ViQuAD1.0/dev_ViQuAD.json \
    --train_dataset_cache_path /Users/phuongnguyen/study/vn-qa-gen/Datasets/output/QG/train_cache.pkl \
    --dev_dataset_cache_path /Users/phuongnguyen/study/vn-qa-gen/Datasets/output/QG/dev_cache.pkl

