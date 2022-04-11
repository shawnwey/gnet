import cv2
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from random import random, choice
from io import BytesIO
from PIL import Image
from PIL import ImageFile
from scipy.ndimage.filters import gaussian_filter


ImageFile.LOAD_TRUNCATED_IMAGES = True

def dataset_folder(opt, root):
    if opt.mode == 'binary':
        return binary_dataset(opt, root)
    if opt.mode == 'filename':
        return FileNameDataset(opt, root)
    raise ValueError('opt.mode needs to be binary or filename.')


def binary_dataset(opt, root):
    composeList = []
    if not opt.isTrain and opt.no_resize:
        pass
    else:
        composeList.append(CustomResize(opt))

    composeList.append(DataAugment(opt))

    if opt.isTrain:
        composeList.append(transforms.RandomCrop(opt.cropSize))
    elif opt.no_crop:
        pass
    else:
        composeList.append(transforms.CenterCrop(opt.cropSize))

    if opt.isTrain and not opt.no_flip:
        composeList.append(transforms.RandomHorizontalFlip())

    composeList.extend([transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ])
    dset = datasets.ImageFolder(root,
                                transforms.Compose(composeList)
                                )
    return dset


class FileNameDataset(datasets.ImageFolder):
    def name(self):
        return 'FileNameDataset'

    def __init__(self, opt, root):
        self.opt = opt
        super().__init__(root)

    def __getitem__(self, index):
        # Loading sample
        path, target = self.samples[index]
        return path


class DataAugment(object):
    def __init__(self, opt):
        self.opt = opt

    def __call__(self, img):
        img = np.array(img)

        if random() < self.opt.blur_prob:
            sig = sample_continuous(self.opt.blur_sig)
            gaussian_blur(img, sig)

        if random() < self.opt.jpg_prob:
            method = sample_discrete(self.opt.jpg_method)
            qual = sample_discrete(self.opt.jpg_qual)
            img = jpeg_from_key(img, qual, method)

        return Image.fromarray(img)


def sample_continuous(s):
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")


def sample_discrete(s):
    if len(s) == 1:
        return s[0]
    return choice(s)


def gaussian_blur(img, sigma):
    gaussian_filter(img[:, :, 0], output=img[:, :, 0], sigma=sigma)
    gaussian_filter(img[:, :, 1], output=img[:, :, 1], sigma=sigma)
    gaussian_filter(img[:, :, 2], output=img[:, :, 2], sigma=sigma)


def cv2_jpg(img, compress_val):
    img_cv2 = img[:, :, ::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:, :, ::-1]


def pil_jpg(img, compress_val):
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format='jpeg', quality=compress_val)
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return img


jpeg_dict = {'cv2': cv2_jpg, 'pil': pil_jpg}


def jpeg_from_key(img, compress_val, key):
    method = jpeg_dict[key]
    return method(img, compress_val)


rz_dict = {'bilinear': Image.BILINEAR,
           'bicubic': Image.BICUBIC,
           'lanczos': Image.LANCZOS,
           'nearest': Image.NEAREST}


class CustomResize(object):
    def __init__(self, opt):  # snr, p 是要传入的多个参数
        self.opt = opt

    def __call__(self, img):  # __call__函数还是只有一个参数传入
        interp = sample_discrete(self.opt.rz_interp)
        return TF.resize(img, self.opt.loadSize, interpolation=rz_dict[interp])
