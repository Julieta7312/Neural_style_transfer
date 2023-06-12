Author: Julieta Matevosyan 

Preparation

cd <nst folder's_directory:root_folder_for_this_README.txt> 
python3 -m venv venv
source venv/bin/activate 
pip install pip_installs.txt

deactivate will deactivate the venv


Object recognition

    - Prepare sample data and fine-tune YOLO. 

    a. Open save_frames.py

    b. Run capture_frames() to download the frames of VIDEO_DIR to IMAGES_PATH. 

    c. Using filter_frames() or del_imgs(), delete the redundant frames in IMAGES_PATH. 
    
    The resulting frames in IMAGES_PATH will be used to fine-tune YOLO. 

    d. Install https://github.com/heartexlabs/labelImg and Label the images as in 
       './to_yolo/dataset/labels'
       python3 ./venv/lib/python3.9/site-packages/labelImg/labelImg.py

    e. Follow https://docs.ultralytics.com/yolov5/tutorials/train_custom_data/ to 
       train the self-labeled data from d.
       After cloning the repository as below,
           git clone https://github.com/ultralytics/yolov5  # clone
           cd yolov5
           pip install -r requirements.txt  # install
       you will probably need something similar to 
           ./to_yolo/dataset.yaml
           ./to_yolo/dataset/images
           ./to_yolo/dataset/labels
        to be copied over to the folder where ./yolov5 is.
        After installing the required packages into Python venv, using that environment to
        reinstall:
            pip uninstall numpy
            pip install numpy
            pip uninstall opencv-python
            pip install opencv-python
            pip uninstall pandas
            pip install pandas

        train the model by running the command:
            python train.py --img 720 --batch 16 --epochs 25 --data dataset.yaml --weights yolov5s.pt
            * train.py is in yolov5. 
        
    f. Save the weight 'last.pt' from step e. to './fine_tuned_mdl/weights/last.pt'.

    g. Run yolo.py to save the image labels predicted by fine-tuned YOLO to IMAGES_PRED_PATH.
       The images and yolo_total_res.csv in
       "./neural_style_transfer/to_yolo/dataset/preds"
       are the results from running yolo.py. ( labeled images and detected object coordinates )

Neural Style Transfer

    a. Open nst.py

    b. Using `# Stylize YOLO detected object. Start ~ End` in nst.py, all the images in 
       './to_yolo/dataset/images' will be read. Then, based on the 
       coordinate and class information of the object (in yolo_total_res.csv), the images will be 
       partially stylized with goldenTexture.jpg or rockTexture.jpg, and saved to OUTPUT_DIR.
       
    c. From `# Explore the extracted features. Start ~ End`, you can extract the features from 
       network layers. I saved the features in 
       "./neural_style_transfer/to_yolo/dataset/layers"



   
