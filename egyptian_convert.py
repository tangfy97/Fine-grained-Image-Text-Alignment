import xml.etree.ElementTree as ET
import os
import numpy as np
from tqdm import tqdm
import json

# this module used to generate caption for each picture and extract features base on the corresponding caption
# parse the xml and get the text for relevant element
def get_text(path):
    tree = ET.parse(path)
    root = tree.getroot()
    file_name = root[1].text
    res = []
    for object in tree.findall("object"):
        res.append(object[0].text)
    return file_name,res

#XML文件解析
# generate features for each image by using parsed xml file
def xml_parser(path,features_path,save_path):

    list_dir = os.listdir(path)
    features = []
    not_list = []
    for image in tqdm(list_dir):
        file_path = os.path.join(features_path, image.split('.')[0] + '.npy')
        feature = np.load(file_path, allow_pickle=True).tolist()["features"]
        print(feature)
        feature = np.array(feature)
        features.append(feature[:10, :])
    #   features = np.concatenate(features)
    features = np.stack(features)
    np.save(save_path, features)
def json_parser(path):
    # load the json files extract the sentence and corresponding picture name.
    with open(path,"r") as file:
        data = json.load(file)
        for image in data["images"]:
            try:
                file_name = image["filename"]+".txt"
                sentence = image["sentences"]
                sentence = sentence[0]["raw"]
            except:
                continue
            # save the sentence in relevant files
            with open(os.path.join("../data/phrase_train", file_name), "w") as f:
                f.write(sentence)
#json_parser("/home/ljd/PycharmProjects/assignment_4_13/image-text/caption_data_pytorch.json")

if __name__ == '__main__':
    xml_parser("../data/phrase_train", "../features_train", "../train_phrase")
    """
    dir = "/egyptian-annotations"
    list_dir = os.listdir(dir)
    list_dir.remove("phrase")lL.' MMMMN
    for path in list_dir:
        print(path)
        file_name, res = get_text(os.path.join(dir,path))
        file_name += ".txt"
        file = dir+"/phrase/"+file_name
        for sen in res:
            with open(file, "w") as f:
                f.write(sen)
    """



