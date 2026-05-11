from setuptools import find_packages, setup

package_name = 'controller'

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
    maintainer='robot',
    maintainer_email='robot@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'motor_control = controller.motor_control:main',
            'a_planner= controller.Aplanner:main',
            'path_track= controller.pathtrack:main',
            'map_test= controller.maptest:main',
            'test= controller.testhsv:main',
            'depth_filter= controller.depthfilterbox:main',
            'depth_scan= controller.deptchscan:main',
            'box= controller.testper:main',
            'box_go= controller.box_go:main',
            'perc= controller.testper:main',
            'cube= controller.cube:main',
            'map= controller.testmapping:main',
            'ring= controller.ring:main',
            'explore= controller.explore:main',
            'task= controller.task:main',
            'obj= controller.obj:main',
        ],
    },
)
