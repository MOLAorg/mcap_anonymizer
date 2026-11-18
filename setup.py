import os
import re

from setuptools import find_packages, setup

package_name = 'mcap_anonymizer'
here = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(here, package_name, '__init__.py')) as f:
    version = re.search(r"__version__ = '([^']+)'", f.read()).group(1)

setup(
    name='mcap-anonymizer',
    version=version,
    packages=find_packages(exclude=['test']),
    package_data={package_name: ['models/*.onnx', 'models/LICENSE-*', '_vendor/mcap/LICENSE']},
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['numpy', 'opencv-python-headless', 'pillow'],
    extras_require={
        'onnxruntime': ['onnxruntime'],
        'compression': ['zstandard', 'lz4'],
        'test': ['pytest'],
    },
    python_requires='>=3.8',
    zip_safe=False,
    author='Jose Luis Blanco-Claraco',
    author_email='joseluisblancoc@gmail.com',
    url='https://github.com/MOLAorg/mcap_anonymizer',
    description='Blurs the faces in the camera images of MCAP files (ROS 2 bags), keeping '
                'everything else unchanged',
    long_description=open(os.path.join(here, 'README.md')).read(),
    long_description_content_type='text/markdown',
    license='BSD-3-Clause',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mcap_anonymize = mcap_anonymizer.cli:main',
        ],
    },
)
