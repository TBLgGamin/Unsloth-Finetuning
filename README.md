# AI Training Program

This program fine-tunes a Llama-3 model using the Unsloth library and saves the fine-tuned model as a Q4 GGUF file.

## Requirements

- Windows Subsystem for Linux (WSL) with Ubuntu 24.04
- Python 3.10
- CUDA-enabled GPU
- Miniconda

## Setup

## Install WSL and Ubuntu

Open your Windows command prompt and run:
   ```bash
   wsl --install Ubuntu-24.04
   ```
Complete the setup by providing a root username and password for Ubuntu.

## Update Ubuntu Packages
Open the Ubuntu terminal (you can access it by typing wsl in the Windows command prompt) and run:
```bash
sudo apt update && sudo apt upgrade
```
## Install Necessary Dependencies
Install essential build tools and Python package manager:
``` bash
sudo apt-get install build-essential gdb python3-pip
```
## Install Miniconda
Install Miniconda for environment management:
```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init
```
Restart your terminal or run
``` bash
source ~/.bashrc
```

## Create and Activate Conda Environment
Create and activate a new Conda environment:
``` bash
conda create --name unsloth_env python=3.10
conda activate unsloth_env
```

## Install Unsloth and Dependencies
Install Unsloth and other required packages:
``` bash
conda install pytorch-cuda=12.1 pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes
pip install colorama
```

## Usage
Run the Training Script
Save the train.py script and run it:
``` bash
python3 train.py
```

## Follow the Prompts
The script will recommend a model based on your system resources.
You can choose to use the recommended model or specify a different one.
Enter the name of the dataset from Hugging Face to use for fine-tuning.
Model Fine-Tuning

The script will load and preprocess the dataset, fine-tune the model, and save the fine-tuned model to the ./fine_tuned_model directory.

Notes
Ensure you have sufficient GPU memory for the model you choose.
Adjust the training parameters in the train.py script as needed.

### Additional Notes:
- Ensure that you have installed all the required dependencies and that your environment is correctly set up.
- If you encounter any issues, consult the [Unsloth GitHub repository](https://github.com/unslothai/unsloth) for more detailed documentation and support.
- Having issues, or want to ask questions about usage? Contact the project maintainer at wenagptmdj@gmail.com