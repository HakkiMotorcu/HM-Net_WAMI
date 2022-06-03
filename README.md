# HM-Net_WAMI

    @article{HMNet,
      title={HM-Net: A Regression Network for Object Center Detection and Tracking on Wide Area Motion Imagery},
      author={Motorcu, Hakki and Ates, Hasan F. and Ugurdag, H. Fatih and Gunturk, Bahadir K.},
      journal={IEEE Access},
      year={2022},
      doi={10.1109/ACCESS.2021.3138980}
    }
    
### Installation
~~~
HM-Net_ROOT=/path/to/clone/HM-Net_ROOT
git clone https://github.com/HakkiMotorcu/HM-Net_WAMI $HM-Net_ROOT
~~~
~~~
cd HM-Net_ROOT
conda create --name HM_Net python=3.9.12
conda activate HM_Net
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
~~~
### Data Format
 ~~~
{HM-Net_ROOT}
  |-- Datasets
  `-- |-- WPAFB
      `-- |--- train
          |    |--- annotations (txt files named as {track_name}.txt  
          |    |--- sequences (folders named after {track_name} contains images 
          |    |--- train.json (coco format annotations can be generated)
          |--- test
          |   |--- annotations
          |   |--- ...
~~~
To generate json files, we provided one example converter under "Source/lib.data/data_tools/" "sat2coco.py", which takes the format below and converts it to coco.
~~~
<frame_index>,<target_id>,<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
~~~
Important Note: All information related to a dataset is kept under "Source/data_conf/" folder. After preparing dataset create a json file (template file can be found on the folder).


### Usage 

For training and testing purposes consult to "Source.lib.setup.py" and given ".sh" files. 

#### Acknowledgement
While writing this repo, we were inspired by the following ifzhang/FairMOT and xingyizhou/CenterTrack repositories. 


