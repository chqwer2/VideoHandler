# activate conda env
# framerate = 10?

source ~/.bashrc
conda activate Stereo

# mk dataset
cd DemoVideo
chmod +x process.sh
./process.sh 'YourVideo'


# visualizing the ITD prediction of videos over time in vis_scripts/vis_video_itd.py. 


# ./scripts/visualization_video.sh 'YourVideo' YOUR_SAVE_PATH