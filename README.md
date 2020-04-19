#### This repo is based on [SCAN](https://github.com/kuanghuei/SCAN) model proposed by Kuang-Huei Lee, Xi Chen, Gang Hua, Houdong Hu and Xiaodong He.

#### [Bottom-up feature repo](https://github.com/violetteshev/bottom-up-features) was used to extract features, pre-trained models were used for extraction tasks.

This repo does NOT contain model files and extracted features, if you want to download them, please use this [link](https://www.dropbox.com/s/5d4d21yro3pp6ws/iamge-text.zip?dl=0) to download the full version (read-to-run) of the project.

### Structure of the project:

* According to captioins and corresponding image labels, we can extract the corresponding image names. This extraction process can be done using `image-text/egyptian_convert.py`, the captions for each image will have a corresponding `.txt` file created. For texting data, these  `.txt` are saved in `phrase` directory and their corresponding image features are saved under `features` directory. After obtained `egptian-test.npy`, the same, we saved the training result under `data/phrase_train`, `features_train` directory, and `data/train_phrase.npy` was obtained for training.

* Use `image-text/preprocess.ipynb` to obtain `data/vocab.json`. We use the API proposed by [Handler et al.](https://github.com/slanglab/phrasemachine) to extract noun phrases.

* For training process, we can simply modify the `PrecompDataset` class in `image-text/data.py` to change related processing methods.

* For testing process, we just need to load our saved model then run `image-text/evaluation.py` script.

* Original datasets of Egyptian arts images were saved under `artworks`.

* For training and testing parameters, check descriptions and instructions in `evaluationi.py` for testing and `train.py` for training.


### Feature extraction command:

```
python bottom-up-features/extract_features.py --image_dir artworks/test --out_dir artworks/features --cfg bottom-up-features/cfgs/faster_rcnn_resnet101.yml --model bottom-up-features/models/bottomup_pretrained_10_100.pth
```
Some remarks:


1. The captions for each image in the original paper were 5 which I changed to 1 in this case
2. Run the following command to train the dataset

```
python train.py --data_name chinese_artworks --logger_name runs/chinese_artworks_scan/log --model_name runs/chinese_artworks_scan/log --max_violation --bi_gru --img_dim 2048
```
3. Pre-trained models are stored under `./runs/` folder