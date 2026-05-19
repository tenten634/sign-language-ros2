from setuptools import setup

package_name = "pipeline_bringup_ros2"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (
            "share/" + package_name + "/launch",
            [
                "launch/pipeline_bringup_ros2.launch.py",
            ],
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Your Name",
    maintainer_email="you@example.com",
    description="ROS2 bringup package for ASL, planner, executor, and evaluation nodes.",
    license="MIT",
    tests_require=["pytest"],
)

