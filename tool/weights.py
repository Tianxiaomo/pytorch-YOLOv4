import hashlib
import logging
import os

from .base import data_dir

from google_drive_downloader import GoogleDriveDownloader as gdd


DARKNET_YOLOV4_WEIGHTS_ID = '1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT'


# Thanks quantumSoup @ https://stackoverflow.com/questions/3431825/generating-an-md5-checksum-of-a-file
def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def validate_checksum(id, dest_path):
    if id == DARKNET_YOLOV4_WEIGHTS_ID:
        return '00a4878d05f4d832ab38861b32061283' == md5(dest_path)

    return False


def download_weights(id=DARKNET_YOLOV4_WEIGHTS_ID):
    dest_path = None
    if id == DARKNET_YOLOV4_WEIGHTS_ID:
        dest_path = os.path.join(data_dir(), 'yolov4.weights')

    if dest_path is None:
        return None

    overwrite = False
    for ii in range(3):
        if ii > 0:
            overwrite = True
            logging.warn("Checksum failed, retrying download...")

        try:
            gdd.download_file_from_google_drive(file_id=id,
                                                dest_path=dest_path,
                                                unzip=True,
                                                showsize=True,
                                                overwrite=overwrite)
        except Exception as e:
            logger.error('Error at %s', 'file download', exc_info=e)
            return None

        if validate_checksum(id, dest_path):
            return dest_path

    logging.error('Unable to validate file checksum')
    return None
