# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
General utils
"""

import contextlib
import glob
import logging
import math
import os
import platform
import random
import re
import shutil
import signal
import time
import urllib
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from subprocess import check_output
from zipfile import ZipFile

import cv2
import numpy as np
import pandas as pd
import pkg_resources as pkg
import torch
import torchvision
import yaml

from utils.downloads import gsutil_getsize
from utils.metrics import box_iou, fitness

# Settings
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
DATASETS_DIR = ROOT.parent / 'datasets'  # YOLOv5 datasets directory
NUM_THREADS = min(8, max(1, os.cpu_count() - 1))  # number of YOLOv5 multiprocessing threads
VERBOSE = str(os.getenv('YOLOv5_VERBOSE', True)).lower() == 'true'  # global verbose mode
FONT = 'Arial.ttf'  # https://ultralytics.com/assets/Arial.ttf

torch.set_printoptions(linewidth=320, precision=5, profile='long')
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})  # format short g, %precision=5
pd.options.display.max_columns = 10
cv2.setNumThreads(0)  # prevent OpenCV from multithreading (incompatible with PyTorch DataLoader)
os.environ['NUMEXPR_MAX_THREADS'] = str(NUM_THREADS)  # NumExpr max threads


def is_kaggle():
    # Is environment a Kaggle Notebook?
    try:
        assert os.environ.get('PWD') == '/kaggle/working'
        assert os.environ.get('KAGGLE_URL_BASE') == 'https://www.kaggle.com'
        return True
    except AssertionError:
        return False


def is_writeable(dir, test=False):
    # Return True if directory has write permissions, test opening a file with write permissions if test=True
    if test:  # method 1
        file = Path(dir) / 'tmp.txt'
        try:
            with open(file, 'w'):  # open file with write permissions
                pass
            file.unlink()  # remove file
            return True
        except OSError:
            return False
    else:  # method 2
        return os.access(dir, os.R_OK)  # possible issues on Windows


def set_logging(name=None, verbose=VERBOSE):
    # Sets level and returns logger
    if is_kaggle():
        for h in logging.root.handlers:
            logging.root.removeHandler(h)  # remove all handlers associated with the root logger object
    rank = int(os.getenv('RANK', -1))  # rank in world for Multi-GPU trainings
    logging.basicConfig(format="%(message)s", level=logging.INFO if (verbose and rank in (-1, 0)) else logging.WARNING)
    return logging.getLogger(name)


LOGGER = set_logging('yolov5')  # define globally (used in train.py, val.py, detect.py, etc.)


def user_config_dir(dir='Ultralytics', env_var='YOLOV5_CONFIG_DIR'):
    # Return path of user configuration directory. Prefer environment variable if exists. Make dir if required.
    env = os.getenv(env_var)
    if env:
        path = Path(env)  # use environment variable
    else:
        cfg = {'Windows': 'AppData/Roaming', 'Linux': '.config', 'Darwin': 'Library/Application Support'}  # 3 OS dirs
        path = Path.home() / cfg.get(platform.system(), '')  # OS-specific config dir
        path = (path if is_writeable(path) else Path('/tmp')) / dir  # GCP and AWS lambda fix, only /tmp is writeable
    path.mkdir(exist_ok=True)  # make if required
    return path


CONFIG_DIR = user_config_dir()  # Ultralytics settings dir


class Profile(contextlib.ContextDecorator):
    # Usage: @Profile() decorator or 'with Profile():' context manager
    def __enter__(self):
        self.start = time.time()

    def __exit__(self, type, value, traceback):
        print(f'Profile results: {time.time() - self.start:.5f}s')


class Timeout(contextlib.ContextDecorator):
    # Usage: @Timeout(seconds) decorator or 'with Timeout(seconds):' context manager
    def __init__(self, seconds, *, timeout_msg='', suppress_timeout_errors=True):
        self.seconds = int(seconds)
        self.timeout_message = timeout_msg
        self.suppress = bool(suppress_timeout_errors)

    def _timeout_handler(self, signum, frame):
        raise TimeoutError(self.timeout_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self._timeout_handler)  # Set handler for SIGALRM
        signal.alarm(self.seconds)  # start countdown for SIGALRM to be raised

    def __exit__(self, exc_type, exc_val, exc_tb):
        signal.alarm(0)  # Cancel SIGALRM if it's scheduled
        if self.suppress and exc_type is TimeoutError:  # Suppress TimeoutError
            return True


class WorkingDirectory(contextlib.ContextDecorator):
    # Usage: @WorkingDirectory(dir) decorator or 'with WorkingDirectory(dir):' context manager
    def __init__(self, new_dir):
        self.dir = new_dir  # new dir
        self.cwd = Path.cwd().resolve()  # current dir

    def __enter__(self):
        os.chdir(self.dir)

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.chdir(self.cwd)


def try_except(func):
    # try-except function. Usage: @try_except decorator
    def handler(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            print(e)

    return handler


def methods(instance):
    # Get class/instance methods
    return [f for f in dir(instance) if callable(getattr(instance, f)) and not f.startswith("__")]


def print_args(name, opt):
    # Print argparser arguments
    LOGGER.info(colorstr(f'{name}: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))


# def init_seeds(seed=0):
    # Initialize random number generator (RNG) seeds https://pytorch.org/docs/stable/notes/randomness.html
    # cudnn seed 0 settings are slower and more reproducible, else faster and less reproducible
    import torch.backends.cudnn as cudnn
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # cudnn.benchmark, cudnn.deterministic = (False, True) if seed == 0 else (True, False)

    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)  # CPU
    # torch.cuda.manual_seed(seed)  # GPU
    # torch.cuda.manual_seed_all(seed)  # All GPU
    # os.environ['PYTHONHASHSEED'] = str(seed)  # 禁止hash随机化
    # torch.backends.cudnn.deterministic = True  # 确保每次返回的卷积算法是确定的
    # torch.backends.cudnn.benchmark = False  # True的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。False保证实验结果可复现

    # torch.manual_seed(seed)  # 固定随机种子（CPU）
    # if torch.cuda.is_available():  # 固定随机种子（GPU)
    #     torch.cuda.manual_seed(seed)  # 为当前GPU设置
    #     torch.cuda.manual_seed_all(seed)  # 为所有GPU设置
    # np.random.seed(seed)  # 保证后续使用random函数时，产生固定的随机数
    # torch.backends.cudnn.deterministic = True  # 固定网络结构

def init_seeds(seed=0, deterministic=False):
    # Initialize random number generator (RNG) seeds https://pytorch.org/docs/stable/notes/randomness.html
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for Multi-GPU, exception safe
    # torch.backends.cudnn.benchmark = True  # AutoBatch problem https://github.com/ultralytics/yolov5/issues/9287
    if deterministic and check_version(torch.__version__,
                                       '1.12.0'):  # https://github.com/ultralytics/yolov5/pull/8213
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        os.environ['PYTHONHASHSEED'] = str(seed)


# def set_seeds(seed):
#     torch.manual_seed(seed)  # 固定随机种子（CPU）
#     if torch.cuda.is_available():  # 固定随机种子（GPU)
#         torch.cuda.manual_seed(seed)  # 为当前GPU设置
#         torch.cuda.manual_seed_all(seed)  # 为所有GPU设置
#     np.random.seed(seed)  # 保证后续使用random函数时，产生固定的随机数
#     torch.backends.cudnn.deterministic = True  # 固定网络结构
# set_seeds(106)




def intersect_dicts(da, db, exclude=()):
    # Dictionary intersection of matching keys and shapes, omitting 'exclude' keys, using da values
    return {k: v for k, v in da.items() if k in db and not any(x in k for x in exclude) and v.shape == db[k].shape}


def get_latest_run(search_dir='.'):
    # Return path to most recent 'last.pt' in /runs (i.e. to --resume from)
    last_list = glob.glob(f'{search_dir}/**/last*.pt', recursive=True)
    return max(last_list, key=os.path.getctime) if last_list else ''


def is_docker():
    # Is environment a Docker container?
    return Path('/workspace').exists()  # or Path('/.dockerenv').exists()


def is_colab():
    # Is environment a Google Colab instance?
    try:
        import google.colab
        return True
    except ImportError:
        return False


def is_pip():
    # Is file in a pip package?
    return 'site-packages' in Path(__file__).resolve().parts


def is_ascii(s=''):
    # Is string composed of all ASCII (no UTF) characters? (note str().isascii() introduced in python 3.7)
    s = str(s)  # convert list, tuple, None, etc. to str
    return len(s.encode().decode('ascii', 'ignore')) == len(s)


def is_chinese(s='人工智能'):
    # Is string composed of any Chinese characters?
    return True if re.search('[\u4e00-\u9fff]', str(s)) else False


def emojis(str=''):
    # Return platform-dependent emoji-safe version of string
    return str.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else str


def file_size(path):
    # Return file/dir size (MB)
    path = Path(path)
    if path.is_file():
        return path.stat().st_size / 1E6
    elif path.is_dir():
        return sum(f.stat().st_size for f in path.glob('**/*') if f.is_file()) / 1E6
    else:
        return 0.0


def check_online():
    # Check internet connectivity
    import socket
    try:
        socket.create_connection(("1.1.1.1", 443), 5)  # check host accessibility
        return True
    except OSError:
        return False


@try_except
@WorkingDirectory(ROOT)
def check_git_status():
    # Recommend 'git pull' if code is out of date
    msg = ', for updates see https://github.com/ultralytics/yolov5'
    s = colorstr('github: ')  # string
    assert Path('.git').exists(), s + 'skipping check (not a git repository)' + msg
    assert not is_docker(), s + 'skipping check (Docker image)' + msg
    assert check_online(), s + 'skipping check (offline)' + msg

    cmd = 'git fetch && git config --get remote.origin.url'
    url = check_output(cmd, shell=True, timeout=5).decode().strip().rstrip('.git')  # git fetch
    branch = check_output('git rev-parse --abbrev-ref HEAD', shell=True).decode().strip()  # checked out
    n = int(check_output(f'git rev-list {branch}..origin/master --count', shell=True))  # commits behind
    if n > 0:
        s += f"⚠️ YOLOv5 is out of date by {n} commit{'s' * (n > 1)}. Use `git pull` or `git clone {url}` to update."
    else:
        s += f'up to date with {url} ✅'
    LOGGER.info(emojis(s))  # emoji-safe


def check_python(minimum='3.6.2'):
    # Check current python version vs. required python version
    check_version(platform.python_version(), minimum, name='Python ', hard=True)


def check_version(current='0.0.0', minimum='0.0.0', name='version ', pinned=False, hard=False, verbose=False):
    # Check version vs. required version
    current, minimum = (pkg.parse_version(x) for x in (current, minimum))
    result = (current == minimum) if pinned else (current >= minimum)  # bool
    s = f'{name}{minimum} required by YOLOv5, but {name}{current} is currently installed'  # string
    if hard:
        assert result, s  # assert min requirements met
    if verbose and not result:
        LOGGER.warning(s)
    return result


@try_except
def check_requirements(requirements=ROOT / 'requirements.txt', exclude=(), install=True):
    # Check installed dependencies meet requirements (pass *.txt file or list of packages)
    prefix = colorstr('red', 'bold', 'requirements:')
    check_python()  # check python version
    if isinstance(requirements, (str, Path)):  # requirements.txt file
        file = Path(requirements)
        assert file.exists(), f"{prefix} {file.resolve()} not found, check failed."
        with file.open() as f:
            requirements = [f'{x.name}{x.specifier}' for x in pkg.parse_requirements(f) if x.name not in exclude]
    else:  # list or tuple of packages
        requirements = [x for x in requirements if x not in exclude]

    n = 0  # number of packages updates
    for r in requirements:
        try:
            pkg.require(r)
        except Exception:  # DistributionNotFound or VersionConflict if requirements not met
            s = f"{prefix} {r} not found and is required by YOLOv5"
            if install:
                LOGGER.info(f"{s}, attempting auto-update...")
                try:
                    assert check_online(), f"'pip install {r}' skipped (offline)"
                    LOGGER.info(check_output(f"pip install '{r}'", shell=True).decode())
                    n += 1
                except Exception as e:
                    LOGGER.warning(f'{prefix} {e}')
            else:
                LOGGER.info(f'{s}. Please install and rerun your command.')

    if n:  # if packages updated
        source = file.resolve() if 'file' in locals() else requirements
        s = f"{prefix} {n} package{'s' * (n > 1)} updated per {source}\n" \
            f"{prefix} ⚠️ {colorstr('bold', 'Restart runtime or rerun command for updates to take effect')}\n"
        LOGGER.info(emojis(s))


def check_img_size(imgsz, s=32, floor=0):
    # Verify image size is a multiple of stride s in each dimension
    if isinstance(imgsz, int):  # integer i.e. img_size=640
        new_size = max(make_divisible(imgsz, int(s)), floor)
    else:  # list i.e. img_size=[640, 480]
        new_size = [max(make_divisible(x, int(s)), floor) for x in imgsz]
    if new_size != imgsz:
        LOGGER.warning(f'WARNING: --img-size {imgsz} must be multiple of max stride {s}, updating to {new_size}')
    return new_size


def check_imshow():
    # Check if environment supports image displays
    try:
        assert not is_docker(), 'cv2.imshow() is disabled in Docker environments'
        assert not is_colab(), 'cv2.imshow() is disabled in Google Colab environments'
        cv2.imshow('test', np.zeros((1, 1, 3)))
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        return True
    except Exception as e:
        LOGGER.warning(f'WARNING: Environment does not support cv2.imshow() or PIL Image.show() image displays\n{e}')
        return False


def check_suffix(file='yolov5s.pt', suffix=('.pt',), msg=''):
    # Check file(s) for acceptable suffix
    if file and suffix:
        if isinstance(suffix, str):
            suffix = [suffix]
        for f in file if isinstance(file, (list, tuple)) else [file]:
            s = Path(f).suffix.lower()  # file suffix
            if len(s):
                assert s in suffix, f"{msg}{f} acceptable suffix is {suffix}"


def check_yaml(file, suffix=('.yaml', '.yml')):
    # Search/download YAML file (if necessary) and return path, checking suffix
    return check_file(file, suffix)


def check_file(file, suffix=''):
    # Search/download file (if necessary) and return path
    check_suffix(file, suffix)  # optional
    file = str(file)  # convert to str()
    if Path(file).is_file() or file == '':  # exists
        return file
    elif file.startswith(('http:/', 'https:/')):  # download
        url = str(Path(file)).replace(':/', '://')  # Pathlib turns :// -> :/
        file = Path(urllib.parse.unquote(file).split('?')[0]).name  # '%2F' to '/', split https://url.com/file.txt?auth
        if Path(file).is_file():
            LOGGER.info(f'Found {url} locally at {file}')  # file already exists
        else:
            LOGGER.info(f'Downloading {url} to {file}...')
            torch.hub.download_url_to_file(url, file)
            assert Path(file).exists() and Path(file).stat().st_size > 0, f'File download failed: {url}'  # check
        return file
    else:  # search
        files = []
        for d in 'data', 'models', 'utils':  # search directories
            files.extend(glob.glob(str(ROOT / d / '**' / file), recursive=True))  # find file
        assert len(files), f'File not found: {file}'  # assert file was found
        assert len(files) == 1, f"Multiple files match '{file}', specify exact path: {files}"  # assert unique
        return files[0]  # return file


def check_font(font=FONT):
    # Download font to CONFIG_DIR if necessary
    font = Path(font)
    if not font.exists() and not (CONFIG_DIR / font.name).exists():
        url = "https://ultralytics.com/assets/" + font.name
        LOGGER.info(f'Downloading {url} to {CONFIG_DIR / font.name}...')
        torch.hub.download_url_to_file(url, str(font), progress=False)


def check_dataset(data, autodownload=True):
    # Download and/or unzip dataset if not found locally
    # Usage: https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128_with_yaml.zip

    # Download (optional)
    extract_dir = ''
    if isinstance(data, (str, Path)) and str(data).endswith('.zip'):  # i.e. gs://bucket/dir/coco128.zip
        download(data, dir=DATASETS_DIR, unzip=True, delete=False, curl=False, threads=1)
        data = next((DATASETS_DIR / Path(data).stem).rglob('*.yaml'))
        extract_dir, autodownload = data.parent, False

    # Read yaml (optional)
    if isinstance(data, (str, Path)):
        with open(data, errors='ignore') as f:
            data = yaml.safe_load(f)  # dictionary

    # Resolve paths
    path = Path(extract_dir or data.get('path') or '')  # optional 'path' default to '.'
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    for k in 'train', 'val', 'test':
        if data.get(k):  # prepend path
            data[k] = str(path / data[k]) if isinstance(data[k], str) else [str(path / x) for x in data[k]]

    # Parse yaml
    assert 'nc' in data, "Dataset 'nc' key missing."
    if 'names' not in data:
        data['names'] = [f'class{i}' for i in range(data['nc'])]  # assign class names if missing
    train, val, test, s = (data.get(x) for x in ('train', 'val', 'test', 'download'))
    if val:
        val = [Path(x).resolve() for x in (val if isinstance(val, list) else [val])]  # val path
        if not all(x.exists() for x in val):
            LOGGER.info('\nDataset not found, missing paths: %s' % [str(x) for x in val if not x.exists()])
            if s and autodownload:  # download script
                root = path.parent if 'path' in data else '..'  # unzip directory i.e. '../'
                if s.startswith('http') and s.endswith('.zip'):  # URL
                    f = Path(s).name  # filename
                    LOGGER.info(f'Downloading {s} to {f}...')
                    torch.hub.download_url_to_file(s, f)
                    Path(root).mkdir(parents=True, exist_ok=True)  # create root
                    ZipFile(f).extractall(path=root)  # unzip
                    Path(f).unlink()  # remove zip
                    r = None  # success
                elif s.startswith('bash '):  # bash script
                    LOGGER.info(f'Running {s} ...')
                    r = os.system(s)
                else:  # python script
                    r = exec(s, {'yaml': data})  # return None
                LOGGER.info(f"Dataset autodownload {f'success, saved to {root}' if r in (0, None) else 'failure'}\n")
            else:
                raise Exception('Dataset not found.')

    return data  # dictionary


def url2file(url):
    # Convert URL to filename, i.e. https://url.com/file.txt?auth -> file.txt
    url = str(Path(url)).replace(':/', '://')  # Pathlib turns :// -> :/
    file = Path(urllib.parse.unquote(url)).name.split('?')[0]  # '%2F' to '/', split https://url.com/file.txt?auth
    return file


def download(url, dir='.', unzip=True, delete=True, curl=False, threads=1):
    # Multi-threaded file download and unzip function, used in data.yaml for autodownload
    def download_one(url, dir):
        # Download 1 file
        f = dir / Path(url).name  # filename
        if Path(url).is_file():  # exists in current path
            Path(url).rename(f)  # move to dir
        elif not f.exists():
            LOGGER.info(f'Downloading {url} to {f}...')
            if curl:
                os.system(f"curl -L '{url}' -o '{f}' --retry 9 -C -")  # curl download, retry and resume on fail
            else:
                torch.hub.download_url_to_file(url, f, progress=True)  # torch download
        if unzip and f.suffix in ('.zip', '.gz'):
            LOGGER.info(f'Unzipping {f}...')
            if f.suffix == '.zip':
                ZipFile(f).extractall(path=dir)  # unzip
            elif f.suffix == '.gz':
                os.system(f'tar xfz {f} --directory {f.parent}')  # unzip
            if delete:
                f.unlink()  # remove zip

    dir = Path(dir)
    dir.mkdir(parents=True, exist_ok=True)  # make directory
    if threads > 1:
        pool = ThreadPool(threads)
        pool.imap(lambda x: download_one(*x), zip(url, repeat(dir)))  # multi-threaded
        pool.close()
        pool.join()
    else:
        for u in [url] if isinstance(url, (str, Path)) else url:
            download_one(u, dir)


def make_divisible(x, divisor):
    # Returns nearest x divisible by divisor
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor


def clean_str(s):
    # Cleans a string by replacing special characters with underscore _
    return re.sub(pattern="[|@#!¡·$€%&()=?¿^*;:,¨´><+]", repl="_", string=s)


def one_cycle(y1=0.0, y2=1.0, steps=100):
    # lambda function for sinusoidal ramp from y1 to y2 https://arxiv.org/pdf/1812.01187.pdf
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1


def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {'black': '\033[30m',  # basic colors
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'bright_black': '\033[90m',  # bright colors
              'bright_red': '\033[91m',
              'bright_green': '\033[92m',
              'bright_yellow': '\033[93m',
              'bright_blue': '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan': '\033[96m',
              'bright_white': '\033[97m',
              'end': '\033[0m',  # misc
              'bold': '\033[1m',
              'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']


def labels_to_class_weights(labels, nc=80):
    # Get class weights (inverse frequency) from training labels
    if labels[0] is None:  # no labels loaded
        return torch.Tensor()

    labels = np.concatenate(labels, 0)  # labels.shape = (866643, 5) for COCO
    classes = labels[:, 0].astype(np.int)  # labels = [class xywh]
    weights = np.bincount(classes, minlength=nc)  # occurrences per class

    # Prepend gridpoint count (for uCE training)
    # gpi = ((320 / 32 * np.array([1, 2, 4])) ** 2 * 3).sum()  # gridpoints per image
    # weights = np.hstack([gpi * len(labels)  - weights.sum() * 9, weights * 9]) ** 0.5  # prepend gridpoints to start

    weights[weights == 0] = 1  # replace empty bins with 1
    weights = 1 / weights  # number of targets per class
    weights /= weights.sum()  # normalize
    return torch.from_numpy(weights)


def labels_to_image_weights(labels, nc=80, class_weights=np.ones(80)):
    # Produces image weights based on class_weights and image contents
    class_counts = np.array([np.bincount(x[:, 0].astype(np.int), minlength=nc) for x in labels])
    image_weights = (class_weights.reshape(1, nc) * class_counts).sum(1)
    # index = random.choices(range(n), weights=image_weights, k=1)  # weight image sample
    return image_weights


def coco80_to_coco91_class():  # converts 80-index (val2014) to 91-index (paper)
    # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
    # a = np.loadtxt('data/coco.names', dtype='str', delimiter='\n')
    # b = np.loadtxt('data/coco_paper.names', dtype='str', delimiter='\n')
    # x1 = [list(a[i] == b).index(True) + 1 for i in range(80)]  # darknet to coco
    # x2 = [list(b[i] == a).index(True) if any(b[i] == a) else None for i in range(91)]  # coco to darknet
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
         35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
         64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
    return x


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
    return y


def xyxy2xywhn(x, w=640, h=640, clip=False, eps=0.0):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right
    if clip:
        clip_coords(x, (h - eps, w - eps))  # warning: inplace clip
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w  # x center
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h  # y center
    y[:, 2] = (x[:, 2] - x[:, 0]) / w  # width
    y[:, 3] = (x[:, 3] - x[:, 1]) / h  # height
    return y


def xyn2xy(x, w=640, h=640, padw=0, padh=0):
    # Convert normalized segments into pixel segments, shape (n,2)
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * x[:, 0] + padw  # top left x
    y[:, 1] = h * x[:, 1] + padh  # top left y
    return y


def segment2box(segment, width=640, height=640):
    # Convert 1 segment label to 1 box label, applying inside-image constraint, i.e. (xy1, xy2, ...) to (xyxy)
    x, y = segment.T  # segment xy
    inside = (x >= 0) & (y >= 0) & (x <= width) & (y <= height)
    x, y, = x[inside], y[inside]
    return np.array([x.min(), y.min(), x.max(), y.max()]) if any(x) else np.zeros((1, 4))  # xyxy


def segments2boxes(segments):
    # Convert segment labels to box labels, i.e. (cls, xy1, xy2, ...) to (cls, xywh)
    boxes = []
    for s in segments:
        x, y = s.T  # segment xy
        boxes.append([x.min(), y.min(), x.max(), y.max()])  # cls, xyxy
    return xyxy2xywh(np.array(boxes))  # cls, xywh


def resample_segments(segments, n=1000):
    # Up-sample an (n,2) segment
    for i, s in enumerate(segments):
        x = np.linspace(0, len(s) - 1, n)
        xp = np.arange(len(s))
        segments[i] = np.concatenate([np.interp(x, xp, s[:, i]) for i in range(2)]).reshape(2, -1).T  # segment xy
    return segments


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2


# ------------------------------------soft nmx------------------------------
def box_iou_for_nms(box1, box2, GIoU=False, DIoU=False, CIoU=False, SIoU=False, EIou=False, eps=1e-7):
    # Returns Intersection over Union (IoU) of box1(1,4) to box2(n,4)

    b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
    w1, h1 = b1_x2 - b1_x1, (b1_y2 - b1_y1).clamp(eps)
    w2, h2 = b2_x2 - b2_x1, (b2_y2 - b2_y1).clamp(eps)

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU or EIou:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        if CIoU or DIoU or EIou:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            elif EIou:
                rho_w2 = ((b2_x2 - b2_x1) - (b1_x2 - b1_x1)) ** 2
                rho_h2 = ((b2_y2 - b2_y1) - (b1_y2 - b1_y1)) ** 2
                cw2 = cw ** 2 + eps
                ch2 = ch ** 2 + eps
                return iou - (rho2 / c2 + rho_w2 / cw2 + rho_h2 / ch2)
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    elif SIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        # SIoU Loss https://arxiv.org/pdf/2205.12740.pdf
        s_cw = (b2_x1 + b2_x2 - b1_x1 - b1_x2) * 0.5 + eps
        s_ch = (b2_y1 + b2_y2 - b1_y1 - b1_y2) * 0.5 + eps
        sigma = torch.pow(s_cw ** 2 + s_ch ** 2, 0.5)
        sin_alpha_1 = torch.abs(s_cw) / sigma
        sin_alpha_2 = torch.abs(s_ch) / sigma
        threshold = pow(2, 0.5) / 2
        sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)
        angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi / 2)
        rho_x = (s_cw / cw) ** 2
        rho_y = (s_ch / ch) ** 2
        gamma = angle_cost - 2
        distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)
        omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2)
        omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)
        shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4)
        return iou - 0.5 * (distance_cost + shape_cost)
    return iou  # IoU


def soft_nms(bboxes, scores, iou_thresh=0.5, sigma=0.5, score_threshold=0.25):
    order = torch.arange(0, scores.size(0)).to(bboxes.device)
    keep = []

    while order.numel() > 1:
        if order.numel() == 1:
            keep.append(order[0])
            break
        else:
            i = order[0]
            keep.append(i)

        iou = box_iou_for_nms(bboxes[i], bboxes[order[1:]], EIou=True).squeeze()

        idx = (iou > iou_thresh).nonzero().squeeze()
        if idx.numel() > 0:
            iou = iou[idx]
            newScores = torch.exp(-torch.pow(iou, 2) / sigma)
            scores[order[idx + 1]] *= newScores

        newOrder = (scores[order[1:]] > score_threshold).nonzero().squeeze()
        if newOrder.numel() == 0:
            break
        else:
            maxScoreIndex = torch.argmax(scores[order[newOrder + 1]])
            if maxScoreIndex != 0:
                newOrder[[0, maxScoreIndex],] = newOrder[[maxScoreIndex, 0],]
            order = order[newOrder + 1]

    return torch.LongTensor(keep)
# ---------------------------------------nmx------------------------------------



def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=(), max_det=300):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """
    # agnostic = True
    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    min_wh, max_wh = 2, 7680  # (pixels) minimum and maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS张震
        # i = NMS(boxes, scores, iou_thres, class_nms='SIoU')
        # i = diou_box_nms(boxes, scores, iou_thres)
        # i = my_soft_nms(boxes, scores, iou_thres)  # SOFT NMS
        # i = soft_nms（boxes, scores, iou_thres）
        # i = NMS1(boxes, scores, iou_thres)
        # i = NMS3(boxes, scores, iou_thres)
        # i = diou_box_nms(boxes, scores, iou_thres)


        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            LOGGER.warning(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output


    # if len(output) >= 2:
    #     a = 0
    #     b = 0
    #     c = 0
    #     for m in range(len(output)):
    #
    #         if output[m][5] == 'shedding_crab':
    #             a += 1
    #         if output[m][5] == 'end_unshell':
    #             b += 1
    #         if output[m][5] == 'crab':
    #             c += 1
    #     if a==1 and b==1 and c==0:
    #         if output[0][5]=='end_unshell':
    #             output = torch.index_select(output, 0, torch.tensor([1]))
    #             output = torch.unsqueeze(output, 0)
    #         elif output[1][5]=='end_unshell':
    #             output = torch.index_select(output, 0, torch.tensor([0]))
    #             output = torch.unsqueeze(output, 0)
    #     elif a==1 and b==0 and c==1:
    #         if output[0][5]=='shedding_crab':
    #             output = torch.index_select(output, 0, torch.tensor([1]))
    #             output = torch.unsqueeze(output, 0)
    #         elif output[1][5]=='shedding_crab':
    #             output = torch.index_select(output, 0, torch.tensor([0]))
    #             output = torch.unsqueeze(output, 0)
    #     elif a==0 and b==1 and c==1:
    #         if output[0][5]=='end_unshell':
    #             output = torch.index_select(output, 0, torch.tensor([1]))
    #             output = torch.unsqueeze(output, 0)
    #         elif output[1][5]=='end_unshell':
    #             output = torch.index_select(output, 0, torch.tensor([0]))
    #             output = torch.unsqueeze(output, 0)
    #
    #     elif a==1 and b==1 and c==1:
    #         if output[0][5]=='end_unshell':
    #             output = torch.index_select(output, 0, torch.tensor([1]))
    #             output = torch.index_select(output, 0, torch.tensor([2]))
    #             output = torch.unsqueeze(output, 0)
    #         elif output[1][5]=='end_unshell':
    #             output = torch.index_select(output, 0, torch.tensor([0]))
    #             output = torch.index_select(output, 0, torch.tensor([2]))
    #             output = torch.unsqueeze(output, 0)
    #         elif output[2][5]=='end_unshell':
    #             output = torch.index_select(output, 0, torch.tensor([0]))
    #             output = torch.index_select(output, 0, torch.tensor([1]))
    #             output = torch.unsqueeze(output, 0)




def strip_optimizer(f='best.pt', s=''):  # from utils.general import *; strip_optimizer()
    # Strip optimizer from 'f' to finalize training, optionally save as 's'
    x = torch.load(f, map_location=torch.device('cpu'))
    if x.get('ema'):
        x['model'] = x['ema']  # replace model with ema
    for k in 'optimizer', 'best_fitness', 'wandb_id', 'ema', 'updates':  # keys
        x[k] = None
    x['epoch'] = -1
    x['model'].half()  # to FP16
    for p in x['model'].parameters():
        p.requires_grad = False
    torch.save(x, s or f)
    mb = os.path.getsize(s or f) / 1E6  # filesize
    LOGGER.info(f"Optimizer stripped from {f},{(' saved as %s,' % s) if s else ''} {mb:.1f}MB")


def print_mutation(results, hyp, save_dir, bucket, prefix=colorstr('evolve: ')):
    evolve_csv = save_dir / 'evolve.csv'
    evolve_yaml = save_dir / 'hyp_evolve.yaml'
    keys = ('metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
            'val/box_loss', 'val/obj_loss', 'val/cls_loss') + tuple(hyp.keys())  # [results + hyps]
    keys = tuple(x.strip() for x in keys)
    vals = results + tuple(hyp.values())
    n = len(keys)

    # Download (optional)
    if bucket:
        url = f'gs://{bucket}/evolve.csv'
        if gsutil_getsize(url) > (evolve_csv.stat().st_size if evolve_csv.exists() else 0):
            os.system(f'gsutil cp {url} {save_dir}')  # download evolve.csv if larger than local

    # Log to evolve.csv
    s = '' if evolve_csv.exists() else (('%20s,' * n % keys).rstrip(',') + '\n')  # add header
    with open(evolve_csv, 'a') as f:
        f.write(s + ('%20.5g,' * n % vals).rstrip(',') + '\n')

    # Save yaml
    with open(evolve_yaml, 'w') as f:
        data = pd.read_csv(evolve_csv)
        data = data.rename(columns=lambda x: x.strip())  # strip keys
        i = np.argmax(fitness(data.values[:, :4]))  #
        generations = len(data)
        f.write('# YOLOv5 Hyperparameter Evolution Results\n' +
                f'# Best generation: {i}\n' +
                f'# Last generation: {generations - 1}\n' +
                '# ' + ', '.join(f'{x.strip():>20s}' for x in keys[:7]) + '\n' +
                '# ' + ', '.join(f'{x:>20.5g}' for x in data.values[i, :7]) + '\n\n')
        yaml.safe_dump(data.loc[i][7:].to_dict(), f, sort_keys=False)

    # Print to screen
    LOGGER.info(prefix + f'{generations} generations finished, current result:\n' +
                prefix + ', '.join(f'{x.strip():>20s}' for x in keys) + '\n' +
                prefix + ', '.join(f'{x:20.5g}' for x in vals) + '\n\n')

    if bucket:
        os.system(f'gsutil cp {evolve_csv} {evolve_yaml} gs://{bucket}')  # upload


def apply_classifier(x, model, img, im0):
    # Apply a second stage classifier to YOLO outputs
    # Example model = torchvision.models.__dict__['efficientnet_b0'](pretrained=True).to(device).eval()
    im0 = [im0] if isinstance(im0, np.ndarray) else im0
    for i, d in enumerate(x):  # per image
        if d is not None and len(d):
            d = d.clone()

            # Reshape and pad cutouts
            b = xyxy2xywh(d[:, :4])  # boxes
            b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # rectangle to square
            b[:, 2:] = b[:, 2:] * 1.3 + 30  # pad
            d[:, :4] = xywh2xyxy(b).long()

            # Rescale boxes from img_size to im0 size
            scale_coords(img.shape[2:], d[:, :4], im0[i].shape)

            # Classes
            pred_cls1 = d[:, 5].long()
            ims = []
            for j, a in enumerate(d):  # per item
                cutout = im0[i][int(a[1]):int(a[3]), int(a[0]):int(a[2])]
                im = cv2.resize(cutout, (224, 224))  # BGR
                # cv2.imwrite('example%i.jpg' % j, cutout)

                im = im[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                im = np.ascontiguousarray(im, dtype=np.float32)  # uint8 to float32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                ims.append(im)

            pred_cls2 = model(torch.Tensor(ims).to(d.device)).argmax(1)  # classifier prediction
            x[i] = x[i][pred_cls1 == pred_cls2]  # retain matching class detections

    return x


def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        path = Path(f"{path}{sep}{n}{suffix}")  # increment path
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory
    return path


# Variables
NCOLS = 0 if is_docker() else shutil.get_terminal_size().columns  # terminal window size for tqdm


# ----------------------------------soft_nms------------------------------------------
def my_soft_nms(bboxes, scores, iou_thresh=0.5, sigma=0.5, score_threshold=0.25):
    bboxes = bboxes.contiguous()

    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    # 计算每个box的面积
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # 首先对所有得分进行一次降序排列,仅此一次,以提高后续查找最大值速度. oeder为降序排列后的索引
    _, order = scores.sort(0, descending=True)
    # NMS后,保存留下来的边框
    keep = []

    while order.numel() > 0:
        if order.numel() == 1:  # 仅剩最后一个box的索引
            i = order.item()
            keep.append(i)
            break
        else:
            i = order[0].item()  # 保留首个得分最大的边框box索引,i为scores中实际坐标
            keep.append(i)
        # 巧妙使用tersor.clamp()函数求取order中当前框[0]之外每一个边框,与当前框[0]的最大值和最小值
        xx1 = x1[order[1:]].clamp(min=x1[i])
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])
        # 求取order中其他每一个边框与当前边框的交集面积
        inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
        # 计算order中其他每一个框与当前框的IoU
        iou = inter / (areas[i] + areas[order[1:]] - inter)  # 共order.numel()-1个

        idx = (iou > iou_thresh).nonzero().squeeze()  # 获取order中IoU大于阈值的其他边框的索引
        if idx.numel() > 0:
            iou = iou[idx]
            newScores = torch.exp(-torch.pow(iou, 2) / sigma)  # 计算边框的得分衰减
            scores[order[idx + 1]] *= newScores  # 更新那些IoU大于阈值的边框的得分

        newOrder = (scores[order[1:]] > score_threshold).nonzero().squeeze()
        if newOrder.numel() == 0:
            break
        else:
            newScores = scores[order[newOrder + 1]]
            maxScoreIndex = torch.argmax(newScores)

            if maxScoreIndex != 0:
                newOrder[[0, maxScoreIndex],] = newOrder[[maxScoreIndex, 0],]
            # 更新order.
            order = order[newOrder + 1]

    # 返回保留下来的所有边框的索引值,类型torch.LongTensor
    return torch.LongTensor(keep)




# score_threshold: 此阈值不同于前面分类置信度阈值
# def my_soft_nms(bboxes, scores, iou_thresh=0.5, sigma=0.5, score_threshold=0.25):
#     bboxes = bboxes.contiguous()
#     x1 = bboxes[:, 0]
#     y1 = bboxes[:, 1]
#     x2 = bboxes[:, 2]
#     y2 = bboxes[:, 3]
#     # 计算每个box的面积
#     areas = (x2 - x1 + 1) * (y2 - y1 + 1)
#     # 首先对所有得分进行一次降序排列,仅此一次,以提高后续查找最大值速度. oeder为降序排列后的索引
#     _, order = scores.sort(0, descending=True)
#     # NMS后,保存留下来的边框
#     keep = []
#
#     while order.numel() > 0:
#         if order.numel() == 1:  # 仅剩最后一个box的索引
#             i = order.item()
#             keep.append(i)
#             break
#         else:
#             i = order[0].item()  # 保留首个得分最大的边框box索引,i为scores中实际坐标
#             keep.append(i)
#         # 巧妙使用tersor.clamp()函数求取order中当前框[0]之外每一个边框,与当前框[0]的最大值和最小值
#         xx1 = x1[order[1:]].clamp(min=x1[i])
#         yy1 = y1[order[1:]].clamp(min=y1[i])
#         xx2 = x2[order[1:]].clamp(max=x2[i])
#         yy2 = y2[order[1:]].clamp(max=y2[i])
#         # 求取order中其他每一个边框与当前边框的交集面积
#         inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
#         # 计算order中其他每一个框与当前框的IoU
#         iou = inter / (areas[i] + areas[order[1:]] - inter)  # 共order.numel()-1个
#
#         idx = (iou > iou_thresh).nonzero().squeeze()  # 获取order中IoU大于阈值的其他边框的索引
#         if idx.numel() > 0:
#             iou = iou[idx]
#             newScores = torch.exp(-torch.pow(iou, 2) / sigma)  # 计算边框的得分衰减
#             scores[order[idx + 1]] *= newScores  # 更新那些IoU大于阈值的边框的得分
#
#         newOrder = (scores[order[1:]] > score_threshold).nonzero().squeeze()
#         if newOrder.numel() == 0:
#             break
#         else:
#             newScores = scores[order[newOrder + 1]]
#             maxScoreIndex = torch.argmax(newScores)
#
#             if maxScoreIndex != 0:
#                 newOrder[[0, maxScoreIndex],] = newOrder[[maxScoreIndex, 0],]
#             # 更新order.
#             order = order[newOrder + 1]
#
#     # 返回保留下来的所有边框的索引值,类型torch.LongTensor
#     return torch.LongTensor(keep)


# def diou_box_nms(self, scores, boxes, iou_thres):
#     if boxes.shape[0] == 0:
#         return torch.zeros(0, device=boxes.device).long()
#     x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
#     areas = (x2 - x1 + 1) * (y2 - y1 + 1)
#     order = torch.sort(scores, descending=True)[1]  # (?,)
#     keep = []
#     while order.numel() > 0:
#         if order.numel() == 1:
#             keep.append(order.item())
#             break
#         else:
#             i = order[0].item()
#             keep.append(i)
#
#             xmin = torch.clamp(x1[order[1:]], min=float(x1[i]))
#             ymin = torch.clamp(y1[order[1:]], min=float(y1[i]))
#             xmax = torch.clamp(x2[order[1:]], max=float(x2[i]))
#             ymax = torch.clamp(y2[order[1:]], max=float(y2[i]))
#
#             inter_area = torch.clamp(xmax - xmin, min=0.0) * torch.clamp(ymax - ymin, min=0.0)
#
#             iou = inter_area / (areas[i] + areas[order[1:]] - inter_area + 1e-16)
#
#             # diou add center
#             # inter_diag
#             cxpreds = (x2[i] + x1[i]) / 2
#             cypreds = (y2[i] + y1[i]) / 2
#
#             cxbbox = (x2[order[1:]] + x1[order[1:]]) / 2
#             cybbox = (y1[order[1:]] + y2[order[1:]]) / 2
#
#             inter_diag = (cxbbox - cxpreds) ** 2 + (cybbox - cypreds) ** 2
#
#             # outer_diag
#             ox1 = torch.min(x1[order[1:]], x1[i])
#             oy1 = torch.min(y1[order[1:]], y1[i])
#             ox2 = torch.max(x2[order[1:]], x2[i])
#             oy2 = torch.max(y2[order[1:]], y2[i])
#
#             outer_diag = (ox1 - ox2) ** 2 + (oy1 - oy2) ** 2
#
#             diou = iou - inter_diag / outer_diag
#             diou = torch.clamp(diou, min=-1.0, max=1.0)
#
#             # mask_ind = (iou <= iou_thres).nonzero().squeeze()
#             mask_ind = (diou <= iou_thres).nonzero().squeeze()
#
#             if mask_ind.numel() == 0:
#                 break
#             order = order[mask_ind + 1]
#     return torch.LongTensor(keep)


def NMS1(boxes, scores, thresholds):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    _, order = scores.sort(0, descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)
        if order.numel() == 1:
            break
        break
    return torch.LongTensor(keep)



# def NMS2(boxes, scores, iou_thresh=0.5):
#     bboxes = boxes.contiguous()
#     x1 = boxes[:, 0].contiguous()
#     y1 = boxes[:, 1].contiguous()
#     x2 = boxes[:, 2].contiguous()
#     y2 = boxes[:, 3].contiguous()
#
#     areas = (x2 - x1 + 1) * (y2 - y1 + 1)
#     _, order = scores.sort(0, descending=True)
#     keep = []
#
#     while order.numel() > 0:
#         if order.numel() == 1:
#             i = order.item()
#             keep.append(i)
#             break
#         else:
#             i = order[0].item()  # 保留首个得分最大的边框
#             keep.append(i)
#         # 巧妙使用tersor.clamp()函数求取每一个边框,与当前框的最大值和最小值
#         xx1 = x1[order[1:]].clamp(min=x1[i])
#         yy1 = y1[order[1:]].clamp(min=y1[i])
#         xx2 = x2[order[1:]].clamp(max=x2[i])
#         yy2 = y2[order[1:]].clamp(max=y2[i])
#         # 求取每一个边框与当前边框的交集面积
#         inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
#         # 计算每一个框与当前框的IoU
#         iou = inter / (areas[i] + areas[order[1:]] - inter)
#         # 保留IoU小于阈值的边框的索引
#         idx = (iou <= iou_thresh).nonzero().squeeze()
#
#         if idx.numel() == 0:
#             break
#         # 这里+1,是为了补充idx与order之间的索引差
#         order = order[idx + 1]
#
#     if len(keep) >= 2:
#         if boxes
#     return torch.LongTensor(keep)



def NMS3(boxes, scores, iou_thresh=0.5):
    bboxes = boxes.contiguous()
    x1 = boxes[:, 0].contiguous()
    y1 = boxes[:, 1].contiguous()
    x2 = boxes[:, 2].contiguous()
    y2 = boxes[:, 3].contiguous()

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    _, order = scores.sort(0, descending=True)
    keep = []

    while order.numel() > 0:
        if order.numel() == 1:
            i = order.item()
            keep.append(i)
            break
        else:
            i = order[0].item()  # 保留首个得分最大的边框
            keep.append(i)
        # 巧妙使用tersor.clamp()函数求取每一个边框,与当前框的最大值和最小值
        xx1 = x1[order[1:]].clamp(min=x1[i])
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])
        # 求取每一个边框与当前边框的交集面积
        inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
        # 计算每一个框与当前框的IoU
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        # 保留IoU小于阈值的边框的索引
        idx = (iou <= iou_thresh).nonzero().squeeze()

        if idx.numel() == 0:
            break
        # 这里+1,是为了补充idx与order之间的索引差
        order = order[idx + 1]

    # 判断keep中边框的类别并进行去重
    class_list = []
    for index in keep:
        class_list.append(boxes[index][-1])

    class_set = set(class_list)
    if len(class_set) == 2 and 0 in class_set and 2 in class_set:
        class_list = [i for i in class_list if i != 2]
    elif len(class_set) == 2 and 1 in class_set and 2 in class_set:
        class_list = [i for i in class_list if i != 2]
    elif len(class_set) == 3 and 0 in class_set and 1 in class_set and 2 in class_set:
        class_list = [i for i in class_list if i not in [0, 2]]

    # 根据去重后的类别重新生成keep列表
    keep = []
    for i, box in enumerate(boxes):
        if box[-1] in class_list and i in keep:
            keep.remove(i)
        elif box[-1] in class_list and i not in keep:
            keep.append(i)

    return torch.LongTensor(keep)




def NMS(boxes, scores, iou_thres, class_nms='CIoU'):
    # class_nms=class_nms
    GIoU=CIoU=DIoU=EIoU=SIoU=False
    if class_nms == 'CIoU':
        CIoU=True
    elif class_nms == 'DIoU':
        DIoU=True
    elif class_nms == 'GIoU':
        GIoU=True
    elif class_nms == 'EIoU':
        EIoU=True
    else :
        SIoU=True
    B = torch.argsort(scores, dim=-1, descending=True)
    keep = []
    while B.numel() > 0:
        index = B[0]
        keep.append(index)
        if B.numel() == 1: break
        iou = bbox_iou(boxes[index, :], boxes[B[1:], :], GIoU=GIoU, DIoU=DIoU, CIoU=CIoU, EIoU=EIoU, SIoU=SIoU)
        inds = torch.nonzero(iou <= iou_thres).reshape(-1)
        B = B[inds + 1]
    return torch.tensor(keep)



class WIoU_Scale:
    ''' if monotonous = None , v1
        if monotonous = True  , v2
        if monotonous = False , v3
    '''

    iou_mean = 1.
    monotonous = True
    _momentum = 1 - 0.5 ** (1 / 7000)
    _is_train = True

    def __init__(self, iou):
        self.iou = iou
        self._update(self)

    @classmethod
    def _update(cls, self):
        if cls._is_train: cls.iou_mean = (1 - cls._momentum) * cls.iou_mean + \
                                         cls._momentum * self.iou.detach().mean().item()

    @classmethod
    def _scaled_loss(cls, self, gamma=1.9, delta=3):
        if isinstance(self.monotonous, bool):
            if self.monotonous:
                return (self.iou.detach() / self.iou_mean).sqrt()
            else:
                beta = self.iou.detach() / self.iou_mean
                alpha = delta * torch.pow(gamma, beta - delta)
                return beta / alpha
        return 1


def bbox_iou(box1, box2, xywh=True,
             GIoU=False,
             DIoU=False,
             CIoU=False,
             SIoU=False,
             EIoU=False,
             Focal=False,
             WIoU=False, scale=False,
             alpha=1, gamma=0.5, eps=1e-7):
    # Returns Intersection over Union (IoU) of box1(1,4) to box2(n,4)

    # Get the coordinates of bounding boxes
    global self
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, (b1_y2 - b1_y1).clamp(eps)
        w2, h2 = b2_x2 - b2_x1, (b2_y2 - b2_y1).clamp(eps)

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps
    if scale:
        self = WIoU_Scale(1 - (inter / union))
    # IoU
    # iou = inter / union # ori iou
    iou = torch.pow(inter / (union + eps), alpha)  # alpha iou
    if CIoU or DIoU or GIoU or EIoU or SIoU or WIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        if CIoU or DIoU or EIoU or SIoU or WIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = (cw ** 2 + ch ** 2) ** alpha + eps  # convex diagonal squared
            rho2 = (((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (
                    b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4) ** alpha  # center dist ** 2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
                with torch.no_grad():
                    alpha_ciou = v / (v - iou + (1 + eps))
                if Focal:
                    # print("你现在使用的是Focal CIoU")
                    return iou - (rho2 / c2 + torch.pow(v * alpha_ciou + eps, alpha)), torch.pow(inter / (union + eps),
                                                                                                 gamma)  # Focal_CIoU
                else:
                    # print("你现在使用的是CIoU")
                    return iou - (rho2 / c2 + torch.pow(v * alpha_ciou + eps, alpha))  # CIoU
            elif EIoU:
                rho_w2 = ((b2_x2 - b2_x1) - (b1_x2 - b1_x1)) ** 2
                rho_h2 = ((b2_y2 - b2_y1) - (b1_y2 - b1_y1)) ** 2
                cw2 = torch.pow(cw ** 2 + eps, alpha)
                ch2 = torch.pow(ch ** 2 + eps, alpha)
                if Focal:
                    # print("你现在使用的是FocalEIoU")
                    return iou - (rho2 / c2 + rho_w2 / cw2 + rho_h2 / ch2), torch.pow(inter / (union + eps),
                                                                                      gamma)  # Focal_EIou
                else:
                    # print("你现在使用的是EIoU")
                    return iou - (rho2 / c2 + rho_w2 / cw2 + rho_h2 / ch2)  # EIou
            elif WIoU:
                # print("你现在使用的是WIoU")
                if scale:
                    return getattr(WIoU_Scale, '_scaled_loss')(self), (1 - iou) * torch.exp(
                        (rho2 / c2)), iou  # WIoU https://arxiv.org/abs/2301.10051
                else:
                    return iou, torch.exp((rho2 / c2))  # WIoU v1
            elif SIoU:
                # SIoU Loss https://arxiv.org/pdf/2205.12740.pdf
                s_cw = (b2_x1 + b2_x2 - b1_x1 - b1_x2) * 0.5 + eps
                s_ch = (b2_y1 + b2_y2 - b1_y1 - b1_y2) * 0.5 + eps
                sigma = torch.pow(s_cw ** 2 + s_ch ** 2, 0.5)
                sin_alpha_1 = torch.abs(s_cw) / sigma
                sin_alpha_2 = torch.abs(s_ch) / sigma
                threshold = pow(2, 0.5) / 2
                sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)
                angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi / 2)
                rho_x = (s_cw / cw) ** 2
                rho_y = (s_ch / ch) ** 2
                gamma = angle_cost - 2
                distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)
                omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2)
                omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)
                shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4)
                if Focal:
                    # print("你现在使用的是FocalSIoU")
                    return iou - torch.pow(0.5 * (distance_cost + shape_cost) + eps, alpha), torch.pow(
                        inter / (union + eps), gamma)  # Focal_SIou
                else:
                    # print("你现在使用的是SIoU")
                    return iou - torch.pow(0.5 * (distance_cost + shape_cost) + eps, alpha)  # SIou
            if Focal:
                # print("你现在使用的是FocalDIoU")
                return iou - rho2 / c2, torch.pow(inter / (union + eps), gamma)  # Focal_DIoU
            else:
                # print("你现在使用的是DIoU")
                return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        if Focal:
            # print("你现在使用的是FocalGIoU")
            return iou - torch.pow((c_area - union) / c_area + eps, alpha), torch.pow(inter / (union + eps),
                                                                                      gamma)  # Focal_GIoU https://arxiv.org/pdf/1902.09630.pdf
        else:
            # print("你现在使用的是GIoU")
            return iou - torch.pow((c_area - union) / c_area + eps, alpha)  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    if Focal:
        # print("你现在使用的是Focal_IoU")
        return iou, torch.pow(inter / (union + eps), gamma)  # Focal_IoU
    else:
        # print("你现在使用的是IoU")
        return iou  # IoU


# ------------------------------------diou-------------------------------
def diou_box_nms(self, scores, boxes, iou_thres):
    if boxes.shape[0] == 0:
        return torch.zeros(0, device=boxes.device).long()
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = torch.sort(scores, descending=True)[1]  # (?,)
    keep = []
    while order.numel() > 0:
        if order.numel() == 1:
            keep.append(order.item())
            break
        else:
            i = order[0].item()
            keep.append(i)

            xmin = torch.clamp(x1[order[1:]], min=float(x1[i]))
            ymin = torch.clamp(y1[order[1:]], min=float(y1[i]))
            xmax = torch.clamp(x2[order[1:]], max=float(x2[i]))
            ymax = torch.clamp(y2[order[1:]], max=float(y2[i]))

            inter_area = torch.clamp(xmax - xmin, min=0.0) * torch.clamp(ymax - ymin, min=0.0)

            iou = inter_area / (areas[i] + areas[order[1:]] - inter_area + 1e-16)

            # diou add center
            # inter_diag
            cxpreds = (x2[i] + x1[i]) / 2
            cypreds = (y2[i] + y1[i]) / 2

            cxbbox = (x2[order[1:]] + x1[order[1:]]) / 2
            cybbox = (y1[order[1:]] + y2[order[1:]]) / 2

            inter_diag = (cxbbox - cxpreds) ** 2 + (cybbox - cypreds) ** 2

            # outer_diag
            ox1 = torch.min(x1[order[1:]], x1[i])
            oy1 = torch.min(y1[order[1:]], y1[i])
            ox2 = torch.max(x2[order[1:]], x2[i])
            oy2 = torch.max(y2[order[1:]], y2[i])

            outer_diag = (ox1 - ox2) ** 2 + (oy1 - oy2) ** 2

            diou = iou - inter_diag / outer_diag
            diou = torch.clamp(diou, min=-1.0, max=1.0)

            # mask_ind = (iou <= iou_thres).nonzero().squeeze()
            mask_ind = (diou <= iou_thres).nonzero().squeeze()

            if mask_ind.numel() == 0:
                break
            order = order[mask_ind + 1]
    return torch.LongTensor(keep)
