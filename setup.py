from setuptools import find_packages, setup
from typing import List

HYPHEN_E_NOT = "-e ."

def get_requirements(file_path: str) -> List[str]:
    requirements = []
    with open(file_path) as file_obj:
        for line in file_obj:
            line = line.strip()
            # Skip empty lines, comments, and the "-e ." flag
            if not line or line.startswith("#") or line == HYPHEN_E_NOT:
                continue
            requirements.append(line)
    return requirements

setup(
    name="CorpusLLM_RAG",
    version="0.0.1",
    author="Uche Nnodim",
    author_email="uchejudennodim@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)
