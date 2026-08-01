import os 
from pathlib import Path
import  logging 

logging.basicConfig(level=logging.INFO, format ='%(asctime)s : %(message)s')

Project_name = "Fraudsentinel"

list_of_files = [
    "requirements.txt",
    "setup.py",
    f"src/{Project_name}/__init__.py",
    f"src/{Project_name}/graph_construction.py",
    f"src/{Project_name}/models.py",
    f"src/{Project_name}/train.py",
    f"src/{Project_name}/evaluate.py",
    f"src/{Project_name}/explain.py",
    "research/trials.ipynb",
    "data/raw/.gitkeep",
    "data/processed/.gitkeep",
]


for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"Creating empty file: {filename}")
    else:
        logging.info(f"{filename} already exists")