import os
from setuptools import setup, find_packages

BASEDIR = os.path.dirname(os.path.abspath(__file__))
VERSION = open(os.path.join(BASEDIR, 'VERSION')).read().strip()

BASE_DEPENDENCIES = [
    'numpy>=1.18.4',
    'torch>=1.4.0',
    'tensorboardX>=2.0',
    'scikit_image>=0.16.2',
    'matplotlib>=2.2.3',
    'tqdm>=4.43.0',
    'easydict>=1.9',
    'Pillow>=7.1.2',
    'opencv_python',
    'wf-pycocotools>=2.0.1',
    'googledrivedownloader>=0.4'
]

# TEST_DEPENDENCIES = [
# ]
#
# DEVELOPMENT_DEPENDENCIES = [
# ]

# Allow setup.py to be run from any path
os.chdir(os.path.normpath(BASEDIR))

setup(
    name='wf-pytorch-yolo-v4',
    packages=find_packages(exclude=('tool.*', 'tool')),
    package_data={'yolov4': ['../cfg/*.cfg', '../data/*.names']},
    version=VERSION,
    include_package_data=True,
    description='Forked version of a minimal PyTorch implementation of YOLOv4',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    original_url='https://github.com/Tianxiaomo/pytorch-YOLOv4',
    original_author='Tianxiaomo',
    url='https://github.com/WildflowerSchools/pytorch-YOLOv4',
    author='Tianxiaomo; Benjamin Jaffe-Talberg',
    author_email='ben.talberg@wildflowerschools.org',
    install_requires=BASE_DEPENDENCIES,
    # tests_require=TEST_DEPENDENCIES,
    # extras_require={
    #     'development': DEVELOPMENT_DEPENDENCIES
    # },
    entry_points={
        "console_scripts": [
            "yolov4 = yolov4:main"
        ]
    },
    keywords=['yolo', 'yolov4', 'pytorch'],
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
    ],
    python_requires=">=3"
)
