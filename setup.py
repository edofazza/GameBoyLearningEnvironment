from setuptools import setup, find_packages

setup(
    name='GameBoyLearningEnvironment',
    version='1.0.1',
    author='Edoardo Fazzari',
    author_email='fazzari.edoardo@gmail.com',
    description='A short description of the package',
    long_description=open('README.md').read(),  # Detailed description, usually from README.md
    long_description_content_type='text/markdown',
    url='https://github.com/edofazza/GameBoyLearningEnvironment',
    packages=find_packages(),  # Automatically finds your package directories
    install_requires=[
        'gymnasium',
        'numpy==1.26.4',
        'pyboy==2.6.0',
        'pygame==2.5.2',
        'pillow==10.3.0',
        'opencv-python==4.10.0.82'
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