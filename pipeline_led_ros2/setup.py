from setuptools import setup

package_name = "pipeline_led_ros2"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Your Name",
    maintainer_email="you@example.com",
    description="LED feedback orchestrator for ASL + planner + executor pipeline states.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "pipeline_led_node = pipeline_led_ros2.pipeline_led_node:main",
        ],
    },
)
