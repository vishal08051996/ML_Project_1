from setuptools import find_packages,setup
from typing import List

Hash_E_DOT = "-e ."

def get_requirements(file_path:str)->List[str]:
    """
    this function returns the requirementsgiven in filepath
    """
    requirements = []
    with open(file_path) as file:
        requirements = file.readlines()
        requirements=[line.replace("\n","") for line in requirements]

    if Hash_E_DOT in requirements:
        requirements.remove(Hash_E_DOT)

    return requirements
    

setup(
    name="ML_Project_1",
    version="0.0.0.1",
    description="This predicts the marks",
    author="Vishal Jadhav",
    packages=find_packages(),
    install_requirements = get_requirements("requirements.txt")
)