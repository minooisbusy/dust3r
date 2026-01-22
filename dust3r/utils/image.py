# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# utilitary functions about images (loading/converting...)
# --------------------------------------------------------
import os
import torch
import numpy as np
import PIL.Image
from PIL.ImageOps import exif_transpose
import torchvision.transforms as tvf
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2  # noqa

INPUT_SIZE = 512

try:
    from pillow_heif import register_heif_opener  # noqa
    register_heif_opener()
    heif_support_enabled = True
except ImportError:
    heif_support_enabled = False

# ToTensor performs reordering from HWC to CHW
ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def img_to_arr( img ):
    if isinstance(img, str):
        img = imread_cv2(img)
    return img

def imread_cv2(path, options=cv2.IMREAD_COLOR):
    """ Open an image or a depthmap with opencv-python.
    """
    if path.endswith(('.exr', 'EXR')):
        options = cv2.IMREAD_ANYDEPTH
    img = cv2.imread(path, options)
    if img is None:
        raise IOError(f'Could not load image={path} with {options=}')
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def rgb_tensor(ftensor, true_shape=None):
    if isinstance(ftensor, list):
        return [rgb_tensor(x, true_shape=true_shape) for x in ftensor]
    if ftensor.ndim == 3 and ftensor.shape[0] == 3:
        ftensor = ftensor.transpose(1, 2, 0)
    elif ftensor.ndim == 4 and ftensor.shape[1] == 3:
        ftensor = ftensor.permute(0, 2, 3, 1)
    if true_shape is not None:
        H, W = true_shape
        ftensor = ftensor[:H, :W]
    if ftensor.dtype == torch.uint8:
        img = torch.float32(ftensor) / 255
    else:
        img = (ftensor * 0.5) + 0.5
    return (img * 255).to(torch.uint8)



def rgb(ftensor, true_shape=None):
    if isinstance(ftensor, list):
        return [rgb(x, true_shape=true_shape) for x in ftensor]
    if isinstance(ftensor, torch.Tensor):
        ftensor = ftensor.detach().cpu().numpy()  # H,W,3
    if ftensor.ndim == 3 and ftensor.shape[0] == 3:
        ftensor = ftensor.transpose(1, 2, 0)
    elif ftensor.ndim == 4 and ftensor.shape[1] == 3:
        ftensor = ftensor.transpose(0, 2, 3, 1)
    if true_shape is not None:
        H, W = true_shape
        ftensor = ftensor[:H, :W]
    if ftensor.dtype == np.uint8:
        img = np.float32(ftensor) / 255
    else:
        img = (ftensor * 0.5) + 0.5
    return img.clip(min=0, max=1)


def _resize_pil_image(img, long_edge_size):
    S = max(img.size)
    if S > long_edge_size:
        interp = PIL.Image.LANCZOS
    elif S <= long_edge_size:
        interp = PIL.Image.BICUBIC
    new_size = tuple(int(round(x*long_edge_size/S)) for x in img.size)
    return img.resize(new_size, interp)


def load_images(folder_or_list, size, square_ok=False, verbose=False):
    """ open and convert all images in a list or folder to proper input format for DUSt3R
    """
    if isinstance(folder_or_list, str):
        if verbose:
            print(f'>> Loading images from {folder_or_list}')
        root, folder_content = folder_or_list, sorted(os.listdir(folder_or_list))

    elif isinstance(folder_or_list, list):
        if verbose:
            print(f'>> Loading a list of {len(folder_or_list)} images')
        root, folder_content = '', folder_or_list

    else:
        raise ValueError(f'bad {folder_or_list=} ({type(folder_or_list)})')

    supported_images_extensions = ['.jpg', '.jpeg', '.png']
    if heif_support_enabled:
        supported_images_extensions += ['.heic', '.heif']
    supported_images_extensions = tuple(supported_images_extensions)

    imgs = []
    for path in folder_content:
        if not path.lower().endswith(supported_images_extensions):
            continue
        img = exif_transpose(PIL.Image.open(os.path.join(root, path))).convert('RGB')
        W1, H1 = img.size
        if size == 224:
            # resize short side to 224 (then crop)
            img = _resize_pil_image(img, round(size * max(W1/H1, H1/W1)))
        else:
            # resize long side to 512
            img = _resize_pil_image(img, size)
        W, H = img.size
        cx, cy = W//2, H//2
        if size == 224:
            half = min(cx, cy)
            img = img.crop((cx-half, cy-half, cx+half, cy+half))
        else:
            halfw, halfh = ((2*cx)//16)*8, ((2*cy)//16)*8
            if not (square_ok) and W == H:
                halfh = 3*halfw/4
            img = img.crop((cx-halfw, cy-halfh, cx+halfw, cy+halfh))

        W2, H2 = img.size
        if verbose:
            print(f' - adding {path} with resolution {W1}x{H1} --> {W2}x{H2}')
        imgs.append(dict(img=ImgNorm(img)[None], true_shape=np.int32(
            [img.size[::-1]]), idx=len(imgs), instance=str(len(imgs))))

    assert imgs, 'no images foud at '+root
    if verbose:
        print(f' (Found {len(imgs)} images)')
    return imgs

CROP = True
NETWORK_INPUT_SIZE = 512
def load_image_with_manipulate(img_path:list, root:str, calib:list, index:int=0, crop_or_resize=CROP):
    # 1. get_calibration
    fx, fy, px, py = calib

    supported_images_extensions = ['.jpg', '.jpeg', '.png']
    if heif_support_enabled:
        supported_images_extensions += ['.heic', '.heif']
    supported_images_extensions = tuple(supported_images_extensions)

    imgs = []
    if not img_path.lower().endswith(supported_images_extensions):
        return

    # img = exif_transpose(PIL.Image.open(os.path.join(root, img_path))).convert('RGB')
    img = imread_cv2(os.path.join(root, img_path), cv2.IMREAD_COLOR_BGR)
    H, W = img.shape[:2]
    cx, cy = W//2, H//2

    if crop_or_resize == CROP:
        if min(H, W) >= NETWORK_INPUT_SIZE:
            # crop a 512x512 window around the principal point (px, py)
            y1, y2 = int(py - 256), int(py + 257)
            x1, x2 = int(px - 256), int(px + 257)
        else:
            isLandscape = W > H
            if isLandscape:
                # 긴 쪽을 512로 scaling
                x1, x2 = int(px - 256), int(px + 257)
                Hin16 = H - H%16
                y1, y2 = int(0), int(Hin16)
            else:
                Win16 = W - W%16
                x1, x2 = int(0), int(Win16)
                y1, y2 = int(px - 256), int(px + 257)

        img_mod = img[y1:y2, x1:x2]

    else:
        # 긴 쪽을 512로 scaling
        scale = NETWORK_INPUT_SIZE/max(H, W)
        new_W, new_H = int(W*scale), int(H*scale)
        img_mod = cv2.resize(img, (new_W, new_H), interpolation=cv2.INTER_LANCZOS4)

    H, W = img_mod.shape[:2]
    K = np.array([[fx, 0, px], [0, fy, py], [0, 0, 1]],dtype=np.float32)

    # ImageNorm: reordering + normalization, [None]: add dimension in front.
    img_dict=dict(img=ImgNorm(img_mod)[None], true_shape=np.int32([[H, W]]), idx=index, instance=str(len(imgs)),
                  camera_intrinsics=K)

    return img_dict # output!!
