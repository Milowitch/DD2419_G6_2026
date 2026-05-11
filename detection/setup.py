from setuptools import find_packages, setup

package_name = 'detection'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='dduberg',
    maintainer_email='danielduberg@gmail.com',
    description='TODO: Package description',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'detection = detection.detection:main',
            'detection2 = detection.detection2:main',
            'detection3 = detection.detection3:main',
            'transformer1 = detection.transformer1:main',
            'transformer2= detection.transformer_stateful:main',
            'box1 = detection.box1:main',
            'box2 = detection.box2:main',
            'box3 = detection.box3:main',
            'box_tf = detection.box3_tf:main',
            'boxtransformer = detection.boxtransformer:main',
            'boxlocal = detection.boxlocal:main'

        ],
    },
)
