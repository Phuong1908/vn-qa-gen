from setuptools import setup, find_packages

setup(
    name="vn_qa_gen",
    version="0.1",
    description="Vietnamese Question Generation",
    packages=find_packages(),
    install_requires=[
        'underthesea>=1.3.0',
        'torch>=1.1.0',
        'tqdm>=4.65.0',
        'transformers>=4.0.0',
        'sentencepiece>=0.1.96',
        'pytorch-ignite>=0.4.0',
        'gdown>=4.7.1',
        'PyDrive2>=1.14.0'
    ],
    python_requires='>=3.7',  # Specify minimum Python version
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    entry_points={
        'console_scripts': [
            'vn-qa-gen=vn_qa_gen.main:main',  # If you want to add command line tools
        ],
    }
)
