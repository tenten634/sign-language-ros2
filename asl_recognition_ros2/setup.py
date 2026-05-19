from setuptools import setup
import os

package_name = 'asl_recognition_ros2'

# Collect model files for data_files
model_base = os.path.join(os.path.dirname(__file__), 'models')
data_files = [
    ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
    ('share/' + package_name, ['package.xml']),
    ('share/' + package_name, ['test_commands.txt']),
]
if os.path.isdir(model_base):
    for root, dirs, files in os.walk(model_base):
        if not files:
            continue
        install_dir = os.path.join('share', package_name, os.path.relpath(root, os.path.dirname(__file__)))
        data_files.append((install_dir, [os.path.join(root, f) for f in files]))

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=data_files,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Robotont Team',
    maintainer_email='robotont@example.com',
    description='ROS2 node for ASL fingerspelling recognition',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'asl_recognition_node = asl_recognition_ros2.asl_recognition_node:main',
        ],
    },
)
