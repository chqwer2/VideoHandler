# https://ificl.github.io/stereocrw/

git clone https://github.com/IFICL/stereocrw

cd stereocrw

source ~/.bashrc
conda env create -f environment.yml
conda activate Stereo


#checkpoint
wget -O pretrained-models/FreeMusic-StereoCRW-1024.pth.tar https://www.dropbox.com/s/qwepkmli4cifn84/FreeMusic-StereoCRW-1024.pth.tar?dl=1

# vis_scripts/eval_itd_in_wild.py
#In params: list_test = 'data/Youtube-ASMR/data-split/keytime/test.csv',


#checkpoint
./scripts/evaluation/evaluation_inthewild.sh


# Preprocessing the video
mkdir Dataset/DemoVideo/RawVideos/YourVideo
cd Dataset/DemoVideo
chmod +x process.sh
./process.sh 'YourVideo'

#Visualization Demo
./scripts/visualization_video.sh 'YourVideo' YOUR_SAVE_PATH
# result in results/YOUR_SAVE_PATH


