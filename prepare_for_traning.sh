file_id=$1

#Download file using gdown
gdown https://drive.google.com/uc?id="$file_id"

#unzip file
unzip model_param.zip

