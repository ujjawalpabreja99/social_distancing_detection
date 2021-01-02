# Social Distancing Detector

## Initial Setup

- Download the dataset from <https://drive.google.com/file/d/1G6nZS-EZLrNBC68CRDf-yo2uj3k32j35/view?usp=sharing> and extract it into the root of the repository in a folder named 'datasets'

## Conda

- We advise creating a conda environment for running this application, the steps for creating one are given below -
- Install conda (You can check the installation by the command "conda --version")
- Creating a new environment with python version 3.7 - `conda create --name "env_name" python=3.7`
- Activating the environment - `conda activate "env_name"`

## Installing the requirements

- Installing the required modules - `pip3 install -r requirements.txt`

## Running the web application

- Start the web server - `python app.py`
- Visit `localhost:5000` to access the web app
- For testing purposes, the app runs on only 5 frames of the video.
- To analyse all the frames of the video, comment out the lines 97, 98 in detect.py
