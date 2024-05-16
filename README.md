
### Requirement
* Python 3.6
* CUDA
* cudnn
* Tensorflow >= 1.0.1
* Pillow(PIL)
* numpy >= 1.12.1
* scipy >= 0.18.1
* imageio
### Step One
Generate train data, you can dirrectly run the command:

python3 preprocess.py

then, images will be generated and saved in './character/'.
### Step Two
Package the images.
Run the command:

python3 package.py

The train data is generated at './data'.
### Step three
Move the folder './data' to an experiment folder.

mkdir exp && mv ./data ./exp/

### Step four
Train your own model

python3 train.py


## Infer
example:

python3 infer.py --model_dir=exp/checkpoint/experiment_0_batch_16 --source_obj=exp/data/val.obj --embedding_ids=x --save_dir=exp/result
```
