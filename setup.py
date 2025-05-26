from setuptools import setup, find_packages

setup(
    name='iffnn',
    version='0.2.0',
    author='Miles Q. Li', 
    author_email='miles.qi.li@mail.mcgill.ca',    # Replace
    description='Interpretable Feedforward Neural Network Library based on Li et al.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/MilesQLi/iffnn', # Replace with your repo URL
    packages=find_packages(),
    install_requires=[
        'torch>=1.7.0', # Specify a reasonable minimum version
        'tqdm>=4.50.0',
        # scikit-learn is only needed for the example, not the core library
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License', # Choose an appropriate license
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.7', # Specify minimum Python version
)