# Vine-Seg Custom Model Training

## Installation

1. Clone the git
   
   ```bash
   git clone https://github.com/s-weissbach/vine-seg_segmentation.git
   ```

2. Create a conda enviorment
   
   ```bash
   conda create -n vineseg_segmentation python=3.9 pip
   conda activate -n vineseg_segmentation
   ```

3. Install the dependencies

   ```bash
   # install dependencies from requirements.txt
   pip install -r requirements.txt
   
   ```

## Data preperation

> [!IMPORTANT] 
> The data preparation notebook is specifically designed to handle the output files generated by Vine-Seg.

To prepare your dataset for training with VineSeg, follow these steps using the provided Jupyter notebook (`data_preperation.ipynb`): 

1. Create Dataset Folder Create a folder within the `data` directory and name it after your dataset. 

2. Create required subfolders (`coco`, `raw`, and `yolo`) in your dataset folder 
   
   
   

3. Copy your images along with the VineSeg generated annotation files in the `raw` folder.
   
   ```bash
   # the resulting folder structure should look like this
   
   data
   ├── coco
   ├── raw
   │   ├── image1.png
   │   ├── image1.json
   │   ├── (...)
   │   └── imageN.json
   └── yolo
   ```

4. In the Jupyter notebook, set the necessary COCO variables:
   
   ```python
   dataset = "example"
   description = "this is a toy example dataset"
   url = "in-house data"
   version = "0.1"
   year = 2024
   contributor = "somebody"
   date_created = "01-01-2024"
   annotation_id_start = 0
   image_fileendings = [".png"]
   ```

5. Run `generate_coco_dict` Function
   
   - You can generate several coco json files and combine into one YOLO data set for training
   
   ```python
   input_folder = f"data/{dataset}/raw"
   coco_output_path = f"data/{dataset}/coco" 
   
   generate_coco_dict(input_path,
                      output_path,
                      description,
                      url,
                      version,
                      year,
                      contributor,
                      date_created,
                      annotation_id_start,
                      image_fileendings,)
   
   ```

6. Use the `coco_seg_to_yolov8` function to convert the generated COCO JSON file(s) to YOLO format.
   
   - adapt the split (train, validation, test) to your need
   
   ```python
   coco_seg_to_yolov8("data/<dataset>/coco", "data/<dataset>/yolo", splits=[0.8, 0.05, 0.15])
   ```

## Training

> [!WARNING]
> We highly recommend to only train on a capable GPU. Training will be very slow and ineffective otherwise.

To train your custom VineSeg model, follow these steps using the provided Jupyter notebook (`training.ipynb`):

1. Chose a model for your training:
   
   - A fresh model, not trained on any calcium imaging data
     
     - ```python
       # model size s
       model = YOLO('yolov8s-seg.pt')
       ```
     
     - Note, there are YOLO models in different sizes. The larger the model, the more computational resources and training data is required. You can load these by sustituting the `s` after the 8 bei (`xs`, `m`, `l`, or `xl`)
   
   - A pre-trained model from Vine-Seg, all stored in the folder `models`. 
     
     - 
       
       ```python
       # pre-trained model on AllenBrain data
       model = YOLO('models/vine_seg-allen.pt')
       ```

2. Train the model
   ```python
   # pre-trained model on AllenBrain data
   model.train(data='data/<dataset>/yolo/data.yaml', epochs=50, imgsz=640, batch=4)
   ```
   - `data`:  The path to the `<dataset>` within the `YOLO` folder.
   - `epochs`: Defines the number of complete passes through the entire training dataset during the training process. In this case, the model will undergo 50 epochs.
   - `imgsz`: Sets the input image size for the training. The images will be resized to a square shape with dimensions `imgsz x imgsz` pixels. Here, the value is set to 640, indicating an image size of 640x640 pixels.
   - `batch`: Specifies the batch size, which is the number of training samples utilized in one iteration. In this case, the batch size is set to 4. Adjusting the batch size can impact training speed and memory requirements.

                

        
