# -*- coding: utf-8 -*-
import librosa
import IPython.display as ipd
import os
import cv2
import random
import torch
import vision
import soundfile
import numpy as np
from PIL import Image
from torchvision import transforms
from docopt import docopt
from torchvision import transforms
from glow.builder import build
from glow.config import JsonConfig


def select_index(name, l, r, description=None):
    index = None
    while index is None:
        print("Select {} with index [{}, {}),"
              "or {} for random selection".format(name, l, r, l - 1))
        if description is not None:
            for i, d in enumerate(description):
                print("{}: {}".format(i, d))
        try:
            line = int(input().strip())
            if l - 1 <= line < r:
                index = line
                if index == l - 1:
                    index = random.randint(l, r - 1)
        except Exception:
            pass
    return index


def run_z(graph, z):
    graph.eval()
    x = graph(z=torch.tensor([z]).cuda(), eps_std=0.3, reverse=True)
    img = x[0].permute(1, 2, 0).detach().cpu().np()
    img = img[:, :, ::-1]
    return img


def save_images(images, names):
    if not os.path.exists("pictures/infer/"):
        os.makedirs("pictures/infer/")
    for img, name in zip(images, names):
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        cv2.imwrite("pictures/infer/{}.png".format(name), img)
        cv2.imshow("img", img)
        cv2.waitKey()


def customized_transform(mel_img, emotion, old_max, old_min, strongness=1.0, hop=512, sr=22050):
    image = mel_img.convert("RGB")
    transform = transforms.Compose([transforms.ToTensor()])
    image = transform(image)
    attr_index = -1
    if emotion == 'SAD':
        attr_index = 5
    elif emotion == 'HAP':
        attr_index = 3
    z_delta = delta_Z[attr_index]
    z_base = graph.generate_z(image)
    layer = 0
    d = z_delta * strongness
    test_img = run_z(graph, z_base + d)
    test_img = np.clip(test_img, 0, test_img.max())
    test_img = np.interp(test_img, (test_img.min(), test_img.max()), (0, 255))
    test_audio = img2audio(test_img[:,:,layer], old_max, old_min, hop_length=hop, sr=sr)
    return test_audio


def img2audio(img,old_max,old_min,hop_length, sr=22050):
    # from 0 - 255 scale to orinally scale
    r1 = reverse_scale_minmax(img,old_max,old_min)
    # inverse log tranformation
    r2 = np.exp(r1)
    # melspectrogram to audio
    r3 = librosa.feature.inverse.mel_to_audio(r2, sr=sr, n_fft=hop_length*2, hop_length=hop_length)
    return r3


def reverse_scale_minmax(X, old_max, old_min):
    return (X/255)*(old_max-old_min)+old_min


def full_cut_by_second(y, sr, name=None):
    n_pieces = len(y)//sr + 1
    result_list = []
    for i in range(1,n_pieces+1):
        start = (i-1)*sr
        if i!= n_pieces:
            end = i*sr
            y_piece = y[start:end]
            result_list.append([y_piece, i])
        else:
            y_piece = y[start:]
            pad_len = sr - len(y_piece)
            y_piece = np.append(y_piece, [0]*pad_len)
            result_list.append([y_piece, i, name])
    return result_list, n_pieces


def audio2img(y, sr, hop_length, n_mels=128):
    # 1.melspectrogram
    mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=hop_length*2, hop_length=hop_length)
    # 2.log-transform.add small number to avoid log(0)
    mels = np.log(mels + 1e-9)
    # 3.standardize into 0 - 255 scale
    img, old_max, old_min = scale_minmax(mels, 0, 255)
    img = img.astype(np.uint8)
    return img, old_max,old_min


def scale_minmax(X, min=0.0, max=255.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled, X.max(), X.min()


if __name__ == "__main__":
    args = docopt(__doc__)
    hparams = args["<hparams>"]
    dataset_root = args["<dataset_root>"]
    z_dir = args["<z_dir>"]
    assert os.path.exists(dataset_root), (
        "Failed to find root dir `{}` of dataset.".format(dataset_root))
    assert os.path.exists(hparams), (
        "Failed to find hparams josn `{}`".format(hparams))
    if not os.path.exists(z_dir):
        print("Generate Z to {}".format(z_dir))
        os.makedirs(z_dir)
        generate_z = True
    else:
        print("Load Z from {}".format(z_dir))
        generate_z = False
    hparams = JsonConfig(hparams)
    dataset = vision.Datasets["cremad"]
    # set transform of dataset
    transform = transforms.Compose([transforms.ToTensor()])
    # build
    graph = build(hparams, False)["graph"]
    dataset = dataset(dataset_root, transform=transform)
    # get Z
    if not generate_z:
        # try to load
        try:
            delta_Z = []
            for i in range(hparams.Glow.y_classes):
                z = np.load(os.path.join(z_dir, "detla_z_{}.npy".format(i)))
                delta_Z.append(z)
        except FileNotFoundError:
            # need to generate
            generate_z = True
            print("Failed to load {} Z".format(hparams.Glow.y_classes))
            quit()
    if generate_z:
        delta_Z = graph.generate_attr_deltaz(dataset)
        for i, z in enumerate(delta_Z):
            np.save(os.path.join(z_dir, "detla_z_{}.npy".format(i)), z)
        print("Finish generating")
    input_dir = '''/content/gdrive/MyDrive/eval_output'''
    output_dir = '''/content/gdrive/MyDrive/eval_output'''
    for i in range(1,6):
        filename = 'orginal_%s.wav'%i
        input_path = os.path.join(input_dir,filename)
        y,sr = librosa.load(input_path,sr=22050)
        y = np.concatenate(([0]*1000, y),axis=None)
        cutted_list, n_pieces = full_cut_by_second(y,int(sr*1.48))
        happy_audio = 0 
        sad_audio = 0
        for row in cutted_list:
            audio = row[0]
            piece_number = row[1]
            img, old_max, old_min = audio2img(audio,sr,hop_length=512,n_mels=128)
            img = Image.fromarray(img)
            happy_piece = customized_transform(img,'HAP',old_max,old_min, strongness=1.2)
            happy_audio = np.concatenate((happy_audio,happy_piece),axis=None)
            sad_piece = customized_transform(img,'SAD',old_max,old_min,strongness=1)
            sad_audio = np.concatenate((sad_audio,sad_piece),axis=None)
        happy_path = os.path.join(output_dir,'%s_%s'%('HAP',i)+'.wav')
        sad_path = os.path.join(output_dir,'%s_%s'%('SAD',i)+'.wav')
        soundfile.write(happy_path, happy_audio, sr)
        soundfile.write(sad_path, sad_audio, sr)
        print('Successful write to %s.'%happy_path)
        print('Successful write to %s.'%sad_path)
