from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = "-e ."
def get_requirements(file_path: str) -> List[str]:
    """
    This function reads the requirements from a file and returns a list of packages.
    """
    requirements = []
    # Open the requirements file and read its contents
    with open(file_path, "r") as file:
        requirements = file.readlines()
    
    # Remove any leading/trailing whitespace characters and filter out empty lines
    requirements = [req.replace("\n","") for req in requirements]
    
    if HYPEN_E_DOT in requirements:
        requirements.remove(HYPEN_E_DOT)
    return requirements

setup(
    name = "mlproject",
    version= "0.1.0",
    author = "Purwa Mugdiya",
    author_email = "purwa.mugdiya@gmail.com",
    packages = find_packages(),
    install_requires = get_requirements("requirements.txt"),
)

#-e . in requirements.txtthis will trigger setup.py to run