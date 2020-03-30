# Growing-Neural-Cellular-Automata
 All credit should be given to the original creators https://distill.pub/2020/growing-ca/

## Dependencies
|  Name |  Version |
|---|---|
| tensorflow |  2.1.0 |
| numpy |  1.18.1 |
| ffmpeg |  4.2 |
| imageio-ffmeg |  0.3.0 |
| node.js |  8.11.1 |

## Setup
    1. Ensure folder train_log and movies exists at root folder 
    2. In Terminal: path to web folder and npm install 

## Guide
1. Find a image file you want to train on
    * png with RGBA format should work 
2. Edit GrowCa.py url variable to point to file.
3. Change EXPERIMENT_TYPE to reflect intended experiment type
4. Run growCa.py
5. (Optional) Run showResultsGrowCa.py to create movie files. 
6. Create the 3 folders in modules_example
    * use_sample_pool_0 damage_n_0 is growing
    * use_sample_pool_1 damage_n_0 is persistence
    * use_sample_pool_1 damage_n_3 is regeneration
    * Drag the 4 files with filename 8000 from train_log into folders 
    * All 3 folders need to exists and contain the files so, if you've just run one experiment copy the files into the folders
7. Modify or add to the EMOJI list in display.py
8. Run display.py
9. Open webgl_models8.zip and drag files into web/webgl_models8
10. Edit emojiList in demo.js and copy target image file into web/images
11. Terminal: Path to web folder and "npm run" 
    


