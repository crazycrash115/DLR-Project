Hello welcome to my readme 

Im pretty sure this is built for windows and if you dont have windows youll need to prob do some bug testing good luck

this is just for me and others to make sure they have everything downloaded right :)

First just pull it from github or download the code, pretty easy 

# Next pip install all the following:

## Base libs (SB3 2.x + Gymnasium + helpers)
pip install -U pip setuptools wheel
pip install "stable-baselines3[extra]>=2.0" gymnasium pygame numpy shimmy

## Flappy Bird environment (Gymnasium-native)
pip install flappy-bird-gymnasium

## Snake environment 
pip install git+https://github.com/sagelywizard/gym_snake.git

not sure might need to get this one too
pip install "gym==0.21.0"

## Pac Man
pip install "gymnasium[atari,accept-rom-license]" "autorom[accept-rom-license]"
python -m AutoROM --accept-license


### How to use

First after downloading all pip stuff go to any training file, its pretty easy just click run, you can change any settings you want but all you need to do out of boot is just run train.

If you want to change the agents rewards check inside wrapper there should be a reward wrapper (some are still being done), when youre happy with your agents training you can look at it with testing make sure to save just incase no idea how VS code deals with new files being added i also just save all.