cd Source
python  main.py --task wami_mot --dataset_conf data_conf/wpafb.json --exp_id wami --model_tag HNetX  --pin_memory True --num_workers 4 --is_validate True --norm_color True --norm_wh False --norm_offset False  --visible_gpus  1,2,3 --gpus 0,1,2 --is_crop True --batch_size 6 --epochs 70 --save_period 2 --reg_weight 0.01 --inp_h 544 --inp_w 960  --lr 0.00013  --track_weight 1 --wh_weight 1 --hm_weight 1
cd ..
