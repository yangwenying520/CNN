import os
import cv2

data_dir = './data'

for phase in ['train', 'test']:
    res = list()
    for folder in os.listdir(os.path.join(data_dir, phase)):
        if '.DS_Store' not in folder:
            for name in os.listdir(os.path.join(data_dir, phase, folder)):
                path = os.path.join(data_dir, phase, folder, name)
                img = cv2.imread(path)
                try:
                    if img.shape not in res:
                        res.append(img.shape)
                except:
                    print(path)
    print('%s images shape: ' % phase, res)
    res.clear()
