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


### Using Terminal (note: these instructions were made with … in mind)
##### Pre-requisites:
Make sure you have python3.9 installed on your device.

#### Step 1: Clone the repository
Start by changing the directory to where you want to download this project. (cd /directory/...) Then run the following:
```
git clone https://github.com/sheepwool/CS123A_TermProject
cd CS123A_TermProject
source packages.sh 
```
(Note: when executing source packages.sh, the Mac terminal may say “command not found: python” if you have a newer version of python. In that case, run this line: alias python=python3)

#### Step 2: Run the code
```
python mushroom_classification.py
```



