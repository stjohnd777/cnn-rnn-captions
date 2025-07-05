### COCO 2014 Dataset Download Instructions

https://cocodataset.org/#home

Here are the instructions to download the COCO 2014 dataset:

1. Create the necessary directories in your project:
```shell script
mkdir -p data/cocoapi/images
mkdir -p data/cocoapi/annotations
```

2. Download the 2014 Training and Validation images [2.5GB + 1.2GB]:
- Training images (13GB): http://images.cocodataset.org/zips/train2014.zip
- Validation images (6GB): http://images.cocodataset.org/zips/val2014.zip

3. Download the 2014 Training/Validation annotations [241MB]:
- http://images.cocodataset.org/annotations/annotations_trainval2014.zip

4. Extract the downloaded files:
```shell script
# Extract images
unzip train2014.zip -d data/cocoapi/images/
unzip val2014.zip -d data/cocoapi/images/

# Extract annotations
unzip annotations_trainval2014.zip -d data/cocoapi/
```


5. Your directory structure should look like this:
```
data/
└── cocoapi/
    ├── images/
    │   ├── train2014/
    │   └── val2014/
    └── annotations/
        ├── instances_train2014.json
        ├── instances_val2014.json
        ├── captions_train2014.json
        └── captions_val2014.json
```

Alternative methods:
1. Using wget:
```shell script
# Download images
wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip

# Download annotations
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
```


2. Using curl:
```shell script
# Download images
curl -O http://images.cocodataset.org/zips/train2014.zip
curl -O http://images.cocodataset.org/zips/val2014.zip

# Download annotations
curl -O http://images.cocodataset.org/annotations/annotations_trainval2014.zip
```

Important Notes:
- Make sure you have enough disk space (around 20GB)
- The downloads might take a while depending on your internet connection
- The extracted data directory should be ignored by git (as we set up in the .gitignore file)
- Verify the downloads completed successfully before extracting
- Keep the zip files until you verify the extraction was successful

Would you like me to provide a script to automate this download process?

---

### Project Overview
A diagram illustrating an image captioning model. It shows a photograph of people shopping at an outdoor market feeding into a deep CNN, which connects to a language generating RNN, outputting the text "A group of people shopping at an outdoor market. There are many vegetables at the fruit stand."
Image captioning model

### Project Overview
In this project, you will create a neural network architecture to automatically generate captions from images.

After using the Microsoft Common Objects in COntext (MS COCO) dataset(opens in a new tab) to train your network, you will test your network on novel images!

Project Instructions
The project is structured as a series of Jupyter notebooks that are designed to be completed in sequential order:

- 0_Dataset.ipynb
- 1_Preliminaries.ipynb
- 2_Training.ipynb
- 3_Inference.ipynb

You can find these notebooks in the Udacity workspace that appears in the concept titled Project: Image Captioning. This workspace provides a Jupyter notebook server directly in your browser.

You can read more about workspaces (and how to toggle GPU support) in the following concept (Introduction to GPU Workspaces). This concept will show you how to toggle GPU support in the workspace.

You MUST enable GPU mode for this project and submit your project after you complete the code in the workspace.

A completely trained model is expected to take between 5-12 hours to train well on a GPU; it is suggested that you look at early patterns in loss (what happens in the first hour or so of training) as you make changes to your model, so that you only have to spend this large amount of time training your final model.

Should you have any questions as you go, please post in the Student Hub!

Evaluation
Your project will be reviewed by a Udacity reviewer against the CNN project rubric(opens in a new tab). Review this rubric thoroughly, and self-evaluate your project before submission. As in the first project, you'll find that only some of the notebooks and files are graded. All criteria found in the rubric must meet specifications for you to pass.

Ready to submit your project?
It is a known issue that the COCO dataset is not well supported for download on Windows, and so you are required to complete the project in the GPU workspace. This will also allow you to bypass setting up an AWS account and downloading the dataset, locally. If you would like to refer to the project code, you may look at the Udacity Image Captioning Project repository for PyTorch 0.4.0 on GitHub(opens in a new tab).

Once you've completed your project, you may only submit from the workspace for this project, see the page: [submit from here] Project: Image Captioning, Pytorch 0.4. Click Submit, a button that appears on the bottom right of the workspace, to submit your project.

For submitting from the workspace, directly, please make sure that you have deleted any large files and model checkpoints in your notebook directory before submission or your project file may be too large to download and grade.

GPU Workspaces
Note: To load the COCO data in the workspace, you must have GPU mode enabled.

In the next section, you'll learn more about these types of workspaces.