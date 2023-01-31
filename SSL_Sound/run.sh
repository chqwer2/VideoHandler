# activate conda env
# framerate = 10?

source ~/.bashrc
conda activate Stereo

# mk dataset
cd DemoVideo
chmod +x process.sh
./process.sh 'YourVideo'

YOUR_SAVE_PATH='/bask/projects/j/jiaoj-3d-vision/Hao/VideoHandler/SSL_Sound/save_path'

# visualizing the ITD prediction of videos over time
./scripts/visualization_video.sh 'YourVideo' $YOUR_SAVE_PATH
