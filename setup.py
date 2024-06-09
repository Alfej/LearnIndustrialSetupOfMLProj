from setuptools import find_packages,setup
from typing import List

HYPHEN_E_DOT = '-e .'

def get_req(file_path:str) -> List[str]:
    '''
    This function will return a list of requirements
    '''
    requr = []
    with open(file_path) as file_obj:
        requr = file_obj.readlines()
        requr = [req.replace("\n","") for req in requr]

        if HYPHEN_E_DOT in requr:
            requr.remove(HYPHEN_E_DOT)

    return requr

setup(
name="LearningMlProj",
version='0.0.1',
author='Alfej',
author_email='alfejmansuri136@gmail.com',
packages=find_packages(),
requires=get_req('requirement.txt')
)