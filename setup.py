from setuptools import setup, find_packages
setup(
    name = "Image Difference",
    version = "1.0",
    packages = find_packages(),
    install_requires = ['mpi4py>=1.3','numpy>=1.6', 'pandas>=0.13'
                        , 'scipy>=0.12', 'cv2>2.3','matplotlib>=1.3'],
    # metadata for upload to PyPI
    author = "Thuso Simon and Valerio Ribeiro",
    author_email = "dr.danger.simon@gmail.com",
    description = "Program to detect birds from video footage",
    license = "MIT",
    keywords = "image differencing, change points",
    url = "https://github.com/drdangersimon/image_diff",   
    
)
