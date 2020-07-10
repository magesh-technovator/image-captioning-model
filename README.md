# Image Captioning model: Streamlit Demo

###  Setup
1. Clone this repository
2. Install requirements: pip install requirements.txt

### Downloading pre-trained model
You can download the pretrained model [here](https://www.dropbox.com/s/ne0ixz5d58ccbbz/pretrained_model.zip?dl=0) and the vocabulary file [here](https://www.dropbox.com/s/26adb7y9m98uisa/vocap.zip?dl=0).

You should extract pretrained_model.zip and move encoder and decoder .pkl to "models/". Vocab file is already placed in "data/" folder.

###  Run without UI
run: python inference.py --image="path/to/image"

### Using Streamlit UI
1. run: streamlit ui.py
2. visit: http://localhost:8501, Upload your image and click on Generate Caption

