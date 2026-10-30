import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'twl_sim'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'modeling'), glob('modeling/*.urdf')),
        (os.path.join('share', package_name, 'modeling/meshes/usd'), glob('modeling/meshes/usd/*.usd')),
        (os.path.join('share', package_name, 'modeling/meshes/stl'), glob('modeling/meshes/stl/*.stl')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='wjddbsgur501',
    maintainer_email='wjddbsgur501@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'mimic_theta = twl_sim.mimic_theta:main',
        ],
    },
)
