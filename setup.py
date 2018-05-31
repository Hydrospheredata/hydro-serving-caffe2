from setuptools import setup, find_packages

setup(
    name="serving-runtime-pytorch",
    version=open("version").read(),
    author="Hydrosphere",
    packages=find_packages(),
    include_package_data=True,
    license="LICENSE.txt",
    install_requires=[
        "numpy>=1.12.3"
    ],
    test_suites=['tests'],
    tests_require=[
        'pytest', 'pylint', 'requests-mock', 'mock>=2.0.0'
    ],
    setup_requires=[
        'pytest-runner',
        "onnx==1.2.*",
        "hydro-serving-grpc==0.1.12"
    ],
    entry_points={
        'console_scripts': [
            'serve-caffe2=caffe2_runtime.main:run'
        ]
    }
)