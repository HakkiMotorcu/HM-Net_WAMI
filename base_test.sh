cd Source
python main.py  --purpose test --max_age 5 --dataset_purp test --num_workers 4 --norm_color True --norm_wh False --norm_offset False --visible_gpus 0 --gpus 0 --model_tag HNetX --det_thres 0.3 --feed_thres 0.3 --save_video True --nms 15 --load_model  ../Experiments/wami_mot/models/model_last.pth  --test_fix_res False  
cd ..
