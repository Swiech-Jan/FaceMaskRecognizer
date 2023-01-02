## PyQt5-Mask-Recognition-ML
**PyQt5 Mask Recognition desktop application using python ML**

## Project Overview:
Desktop application that uses default camera to recognize if person wears mask or not.
Application is able to verify if mask is wore properly - verification if mask is wore partially on chin or only nose.

### Dataset used in project:

[Cabani Masket Face](https://github.com/cabani/MaskedFace-Net)
**Only a subset of pictures is used to train the model**

### Data split:
**Train: 80%**
**Test: 20%**

### CNN is used for face and mask detection.

### Libraries used in project:
- PyQt5 For frontend Desktop app
- numpy
- pandas
- scipy
- pillow
- opencv-python For image-face recognition
- matplotlib - data visualization to help when training the model
- scikit-learn
- jupyter notebook - work environment for ML part
- tensorflow - for Models

## Running the application:
### Open project root folder and type the following command in the console:
- python app.py
