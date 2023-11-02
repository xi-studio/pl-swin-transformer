 python train.py --cfg configs/swinv2_base_patch4_window8_256.yaml --accelerator gpu --devices -1 --max_epochs 2

ffmpeg -f image2 -i combine/wind_u_%03d.png output.mp4
