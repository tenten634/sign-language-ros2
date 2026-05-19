from setuptools import setup

package_name = "pipeline_eval_ros2"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools", "psutil"],
    zip_safe=True,
    maintainer="Your Name",
    maintainer_email="you@example.com",
    description="ROS2 evaluation package for ASL to LLM to execution pipeline metrics.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "pipeline_eval_ros2 = pipeline_eval_ros2.pipeline_eval_ros2:main",
            "pipeline_eval_summary = pipeline_eval_ros2.summarize_pipeline_eval:main",
        ],
    },
)
