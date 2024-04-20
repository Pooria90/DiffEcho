# Unconditional Generation Task

## Unconditional Model Training
To train the unconditional model, navigate to the unconditional directory and execute the following command:
```bash
python train_unconditional.py --class_ultra <class_unltrasound> --num_epochs 400
```
Also to use Debian systems(like Sockeye srever we put the) there is a script.
- Note the class name is one the following options:
    - ch2_ed
    - ch2_es
    - ch4_ed
    - ch4_es

## Data Generation
To generate data using the trained models, run:

```bash
python data_generation.py --number_of_images 200
```
Also, you can use data_generation.sh in this directory. if you are working with Sockeye(Debian servers).
Generated images will be saved in the generated folder.

