from setuptools import setup, find_packages

setup(
    name='GameBoyLearningEnvironment',  # Replace with your package name
    version='0.1.4',  # Initial release version
    author='Edoardo Fazzari',
    author_email='edoardo.fazzari@gmail.com',
    description='A short description of the package',
    long_description=open('README.md').read(),  # Detailed description, usually from README.md
    long_description_content_type='text/markdown',
    url='https://github.com/edofazza/GameBoyLearningEnvironment',
    packages=find_packages(),  # Automatically finds your package directories
    install_requires=[
        'gymnasium',
        'numpy',
        'pyboy==1.6.14'
    ],
    package_data={
        'gle': ['roms/*'],  # Include all files in the data directory
    },
    include_package_data=True,
    classifiers=[  # Optional: metadata for searchability
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.11',
)