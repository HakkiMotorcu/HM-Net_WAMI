import os
import numpy as np
import json
import cv2




def vis2coco(split,DATA_PATH,show=False):
    data_path = DATA_PATH  + '/sequences'
    out_path = DATA_PATH + '/{}.json'.format(split)
    out = {'images': [], 'annotations': [],
           'categories': [{'id': 0, 'name': 'ignore'},
                          {'id': 1, 'name': 'pedestrain'},
                          {'id': 2, 'name': 'people'},
                          {'id': 3, 'name': 'bicycle'},
                          {'id': 4, 'name': 'car'},
                          {'id': 5, 'name': 'van'},
                          {'id': 6, 'name': 'truck'},
                          {'id': 7, 'name': 'tricycle'},
                          {'id': 8, 'name': 'awning-tricycle'},
                          {'id': 9, 'name': 'bus'},
                          {'id': 10, 'name': 'motor'},
                          {'id': 11, 'name': 'others'},
                          ], 'videos': []}
    seqs = os.listdir(data_path)
    image_cnt = 0
    ann_cnt = 0
    video_cnt = 0
    #pdb.set_trace()
    for seq in sorted(seqs):
      video_cnt += 1
      out['videos'].append({
        'id': video_cnt,
        'file_name': seq})
      seq_path = '{}/{}/'.format(data_path, seq)
      img_path = seq_path
      ann_path = DATA_PATH + '/annotations/'+seq+'.txt'
      images = sorted(os.listdir(img_path))
      num_images = len([image for image in images if 'jpg' in image])
      image_range = [0, num_images - 1]
      for i in range(num_images):
        if (i < image_range[0] or i > image_range[1]):
          continue
        image_info = {'file_name': '{}/{}'.format(seq, images[i]),
                      'id': image_cnt + i + 1,
                      'frame_id': i + 1 - image_range[0],
                      'prev_image_id': image_cnt + i if i > 0 else -1,
                      'next_image_id': \
                        image_cnt + i + 2 if i < num_images - 1 else -1,
                      'video_id': video_cnt}
        out['images'].append(image_info)
      if show:
        print('{}: {} images'.format(seq, num_images))
      if split != 'challenge':
        anns = np.loadtxt(ann_path, dtype=np.float32, delimiter=',')

        if show:
          print(' {} ann images'.format(int(anns[:, 0].max())))
        for i in range(anns.shape[0]):
          frame_id = int(anns[i][0])
          if (frame_id - 1 < image_range[0] or frame_id - 1> image_range[1]):
            continue
          track_id = int(anns[i][1])
          cat_id = int(anns[i][7])
          ann_cnt += 1

          ann = {'id': ann_cnt,
                 'category_id': cat_id,
                 'image_id': image_cnt + frame_id,
                 'track_id': track_id,
                 'bbox': anns[i][2:6].tolist(),
                 'conf': float(anns[i][6])}
          out['annotations'].append(ann)
      image_cnt += num_images
    print('loaded {} for {} images and {} samples'.format(
      split, len(out['images']), len(out['annotations'])))
    json.dump(out, open(out_path, 'w'))
