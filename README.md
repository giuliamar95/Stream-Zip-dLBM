# Stream-Zip-dLBM
Deep dynamic co-clustering of count data streams: application to pharmacovigilance


## Execution Instructions

To execute the code, follow these steps:

1. Clone this repository to your local machine:

`git clone https://github.com/giuliamar95/Stream-Zip-dLBM.git`

Navigate to the cloned repository:

`cd Stream-Zip-dLBM`

Create a Conda virtual environment with R and Reticulate:

` conda create -n r-reticulate` 

Activate the Conda virtual environment:

` conda activate r-reticulate `

Install PyTorch and its dependencies:

` conda install pytorch::pytorch torchvision torchaudio -c pytorch `

Install additional Python dependencies:

` conda install numpy matplotlib pandas math random scipy `

Execute the R script called `ScriptStream_Zip-dLBM.R` to run the main function.
