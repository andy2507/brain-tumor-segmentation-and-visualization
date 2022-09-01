import csv
from flask import Flask, request, render_template, url_for, redirect
from segresnet import *
import os
from classification import *
from yolo import *
from normalize import *

app = Flask(__name__)


port = 5100
filename = ""
yolo_results = ""
tumor_type=""
csv_file=""
yolo_file=""
segmentation=""

print("API running on port : {} ".format(port))

@app.route('/')
def my_form():
    return render_template('index.html')

@app.route('/load')
def load_file():
    return render_template('upload.html')

@app.route('/load', methods=['POST'])
def upload():
    files = request.files.getlist('file[]')
    file_name = os.path.split(files[0].filename)[0]
    global filename
    filename=file_name
    if file_name in os.listdir():
        return render_template('index.html', filename=file_name+" already loaded!")
    os.mkdir(file_name)
    os.mkdir(os.path.join(file_name,'results'))
    for file in files:
        file.save(file.filename)
    return render_template('index.html', filename=file_name+" loaded successfully!")


@app.route('/segmentation')
def segmentation_run():
    global filename,tumor_type,csv_file,yolo_file,segmentation
    segmentation = get_pred(filename)
    return render_template('index.html',filename=filename,tumor_type=tumor_type,csv_file=csv_file,yolo_file=yolo_file,segmentation=segmentation)

def get_pred(file_name):
    predict_with_segresnet = PredictWithSegResNet(file_name)
    y_pred = predict_with_segresnet.get_y_pred()
    seg = predict_with_segresnet.get_seg()
    flair_img = predict_with_segresnet.get_flair()
    flair_img = np.rot90(flair_img,1,axes=(1,2))
    # predict_with_segresnet.show_slice(flair_img,y_pred,seg)
    predict_with_segresnet.save_nifti(y_pred)
    return predict_with_segresnet.return_dest()

@app.route('/yolo')
def load_file_yolo():    
    global filename,tumor_type,csv_file,yolo_file,segmentation
    if filename not in os.listdir():
        return render_template("404 error")
    yolo_results=predict_with_yolo(filename)
    csv_file=yolo_results[0]
    yolo_file=yolo_results[1]
    return render_template('index.html',filename=filename,tumor_type=tumor_type,csv_file=csv_file,yolo_file=yolo_file,segmentation=segmentation)


def predict_with_yolo(filename):
    prediction = PredictionWithYolo(filename)
    prediction.slices()
    prediction.get_results()
    return prediction.return_dest()

@app.route('/classify')
def load_file_classify():
    global filename,tumor_type,csv_file,yolo_file,segmentation
    # if file_name in os.listdir():
    #     tumor_type = classify_tumor_using_densenet(file_name)
    #     return render_template("classify.html",tumor_type=tumor_type)
    # os.mkdir(file_name)
    # os.mkdir(os.path.join(file_name,'results'))
    # for file in files:
    #     file.save(file.filename)
    tumor_type = classify_tumor_using_densenet(filename)
    return render_template("index.html",filename=filename,tumor_type=tumor_type,csv_file=csv_file,yolo_file=yolo_file,segmentation=segmentation)

    

def classify_tumor_using_densenet(filename):
    prediction = PredictUsingDensenet()
    class_name = prediction.predict(filename)
    return class_name

@app.route('/run_all')
def run_all():
    global filename,tumor_type,csv_file,yolo_file,segmentation    
    tumor_type = classify_tumor_using_densenet(filename)
    yolo_results = predict_with_yolo(filename)
    segmentation_results = get_pred(filename)
    csv_file=yolo_results[1]
    yolo_file=yolo_results[0]
    segmentation=segmentation_results
    return render_template("index.html",filename=filename,tumor_type=tumor_type,csv_file=csv_file,yolo_file=yolo_file,segmentation=segmentation)



@app.route('/yolo_table/<file_name>')
def table(file_name):    
    # data = pd.read_csv(file_name)
    data = pd.read_csv(os.path.join('/home/andrea/Notebooks/FYP/frontend_html_css',file_name,'results','prediction_yolo_bboxes.csv'))
    return render_template('csv_table.html', tables=[data.to_html()], titles=[''])

@app.route('/normalize_frame')
def normalize_frame():    
    global filename,tumor_type,csv_file,yolo_file,segmentation 
    normalize = Normalize(filename)
    normalize.normalize_frame()
    return render_template("index.html",filename=f'Normalized by frame file of {filename} saved!',tumor_type=tumor_type,csv_file=csv_file,yolo_file=yolo_file,segmentation=segmentation)

@app.route('/normalize_volume')
def normalize_volume():    
    global filename,tumor_type,csv_file,yolo_file,segmentation 
    normalize = Normalize(filename)
    normalize.normalize_volume()
    return render_template("index.html",filename=f'Normalized by volume file of {filename} saved!',tumor_type=tumor_type,csv_file=csv_file,yolo_file=yolo_file,segmentation=segmentation)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=port, debug=True)