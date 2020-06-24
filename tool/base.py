import os
import platform


# data_dir_default & data_dir inspired by mxnet's method of downloading and caching models: https://github.com/apache/incubator-mxnet/blob/master/python/mxnet/base.py
def data_dir_default():
    """
    :return: default data directory depending on the platform and environment variables
    """
    system = platform.system()
    if system == 'Windows':
        return os.path.join(os.environ.get('APPDATA'), 'yolov4')
    else:
        return os.path.join(os.path.expanduser("~"), '.yolov4')


def data_dir():
    """
    :return: data directory in the filesystem for storage, for example when downloading models
    """
    return os.getenv('YOLOV4_HOME', data_dir_default())
