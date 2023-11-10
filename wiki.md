 python train.py --cfg configs/swinv2_base_patch4_window8_256.yaml --accelerator gpu --devices -1 --max_epochs 2

ffmpeg -f image2 -i combine/wind_u_%03d.png output.mp4

python era5_train.py --cfg configs/tp_512.yaml --accelerator cpu --devices 1 --max_epochs 2
python era5_train.py --cfg configs/tp_512.yaml --accelerator tpu --devices 4 --max_epochs 2 --batch-size 32
