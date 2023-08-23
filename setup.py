from setuptools import setup, find_packages

setup(
    name='data_preprocess',  # 您的 Python 库的名称
    version='0.1.0',  # 版本号
    description='A collection of data preprocessing functions',  # 描述
    packages=find_packages(),  # 您的 Python 库的所有包
    install_requires=[  # 您的 Python 库所需要的依赖项
        'numpy',
        'filterpy',
        'statsmodels',
        'pywt',
        'pandas',
        'sklearn',
        'scipy',
        'PyEMD'
    ],
    python_requires='>=3.5'  # Python 版本要求
)
