# CS123A Term Project: Mushroom Classification Using a Multi-Layer Perceptron Model

Author: Dana Shakrovsky

Creation Date: March 25, 2025

### Project Description
This module builds a simple multi-layer perceptron to classify simulated mushrooms (UC Irvine "Secondary Mushroom" dataset) into two categories: edible or poisonous.
Additionally, it also builds a single-layer perceptron to compare the performance of the two perceptrons to each other. This code is written in a Google Colaboratory notebook using python.

## How to Run
To view the code and outputs of the original run, see the file <b>CS123A_TermProject_DanaShakrovsky.ipynb</b>

To run the code, there are three options: 

### Running in Google Colaboratory
1. Navigate to the colab notebook: https://colab.research.google.com/github/sheepwool/CS123A_TermProject/blob/main/CS123A_TermProject_DanaShakrovsky.ipynb
2. Go to File >  Save a copy in drive
3. Close the original, go to the copy
4. Go to Runtime > Run all

### Using an IDE (such as PyCharm or VisualStudio)

1. Clone the repository (should see a button saying “Clone Repository”)
    1. Enter the url https://github.com/sheepwool/CS123A_TermProject where indicated
2. Open the terminal and run source packages.sh 
3. Run the program either by clicking the run button or though the terminal: python mushroom_classification.py


### Using Terminal (Windows OS)
##### (note: these instructions were created and tested for Windows, therefore it is not guaranteed these will work for any other OS)
#### Pre-requisites:
Make sure you have <b>python3.9</b> installed on your device.

#### Step 1: Clone the repository and prepare venv
Start by changing the directory to where you want to download this project. (cd /directory/...) Then run the following:
```
git clone https://github.com/sheepwool/CS123A_TermProject
cd CS123A_TermProject
pythonenv.cmd

```
After completing the above steps, you should now have entered the virtual python environment <b>venv</b>.

#### Step 2: Installing packages
Execute the following commands in venv:
```
pip install ucimlrepo
pip install scikit-learn
pip install pandas
pip install tensorflow
pip install numpy
pip install matplotlib

```

#### Step 3: Run the code
```
python mushroom_classification.py
```



