from setuptools import setup

package_name = "llm_planner_ros2"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    package_data={
        package_name: [
            "robot_capabilities.json",
            "locations.json",
        ],
    },
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Your Name",
    maintainer_email="you@example.com",
    description="ROS2 nodes that use an embedded llm_planner to convert recognized sign text into motion plans and execute them.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "llm_planner_nav_node = llm_planner_ros2.llm_planner_nav_node:main",
            "llm_plan_executor_node = llm_planner_ros2.plan_executor_node:main",
        ],
    },
)

