file_id=$1
file_name=$2

#Download file using gdown
gdown https://drive.google.com/uc?id="$file_id"

#unzip file
unzip $file_name

