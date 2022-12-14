# Brain Tumor Segmentationa and Visualization

## Visualizer
The visualizer was designed using HTML and CSS. The Papaya visualizer is open sourced from <insert link>

## Segmentation 
A 3D segmentation model is used to obtain the segmentation of tumor tissues from normal tissues

## Tumor Localization
A YOLOv5 model is used to localize the areas of the tumor regions

## Tumor Classification
A 3D Classification model is used to classify the tumor regions into the respective classes

## Running the models
Clone the repository using 

`git clone https://github.com/andy2507/brain-tumor-segmentation-and-visualization.git`

## Loading the models
You will have to update the `root_directory` and `ckpt_path` parameters in the files `classification.py`,`segresnet.py` and `yolo.py` to run the app successfully. The models can be found in https://drive.google.com/drive/folders/1yAjUVtIYICDGq4j9hh4IuUEkkKFgyRK1?usp=sharing. Download the files and map the directories accordingly. The root_directory will be the directory the folder is present in  

## Running the models
First enter the repository folder and install the dependencies using `pip install requirements.txt`
To run, while you are still inside the repository folder, type `python app.py` into the command line. 

Go to the link provided and load a folder containing all four modalities: flair, T1, T1ce T2 using the option in File. Run the models necessary and view the results on the visualizer! Two examples along with the results are present in the repository as well!
