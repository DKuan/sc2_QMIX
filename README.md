# QMIX for StarCraft II
## No. of Edition: 12-16-14-26 
The edition of the project uploaded is 12-18-14-22, which has been updated to speed the process of training. This edition have the same effect as the paper saies.

## Description:
This project is created for using QMIX algorithm in StarCraft II mini-games. The units in these maps are partial-obs, which will show the advantage of the QMIX.   

## Installation
known dependencies: Python(3.6.8), StarCraft(3.16.1 or latest version), Pysc2(3.0.0), smac(0.1.0), Pytorch(1.1.0), Numpy(1.17.3)
Step 1: Install the StarCraft and Pysc2 from the website:OpenAI Gym(0.10.5://github.com/deepmind/pysc2)   
Step 2: Install the environment of smac(a modified env for StarCraft II based on Pysc2) from https://github.com/oxwhirl/smac   
Step 3: Change the Hyper-parameters of the algorithm in 'arguments.py'   
Step 4: Run the main.py file with 'python main.py --map_name=8m --per_episode_max_len=80'   
Step 5: Enjoy the model trained by yourself, you can run the command "python enjoy.py --map_name=1c3s5z --old_model_name=./models/1912_190153/"    
  
## Code Structure(update in the future):
./main.py main function of the project    
./enjoy.py  the file to test the model    
./model.py  define of the models     
./arguments.py  the hyper-pars for the project    
./replay_buffer.py  the memory of the agent    
./Q_MIX.py   the class for the agent of QMIX     

## Command line options
### Environment options 
--map_name: defines which environment in the MPE is to be used (default: "simple")   
--per_episode_max_len: maximum length of each episode for the environment (default: It depends on the map that you are training, see from the papers )     
--max_episode: total number of training episodes (default: 150000)   

### Training options
--lr: learning rate for Adam optimizer(default: 5e-4)   
--gamma: discount factor (default: 0.99)   
--batch_size: batch size (default: 32 episodes)   

### Checkpointing
--save_dir: directory where intermediate training results and model will be saved (default: "/models")    
--fre4save_model: model is saved every time this number of game episodes has been completed (default: 234)   
--start_save_model: the time when we start to save the model(default: 800)   

## Link for blog
DeepMind Github: https://github.com/deepmind/pysc2    
Another QMIX algorithm:: https://github.com/starry-sky6688/StarCraft   
The explorer of Game AI: https://zhuanlan.zhihu.com/c_186658689   
