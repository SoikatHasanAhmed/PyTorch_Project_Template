# PyTorch Project template with/without GAN framework

The aim of this repo is to create my personal PyTorch project framework.

Most of the time, whenever I am working on a new project, I either work on my previous template (which is good but not
optimized and doesn't follow oop design) or write everything from scratch. So, after being lazy for over one-and-a-half
year, it's high time I focused on making my own project framework.

Some things are needed to be said beforehand:

- *This work is currently undergoing trials and errors. I haven't checked everything yet. I just took the code and
  started editing. I have to make sure everything works before finalizing the template. I will make some example models
  and test with this framework.*
- I don't like json format. Mainly because I can't comment, and I don't like the look of json files. I apologize for
  being a language-phobic. It is what it is. I prefer YAML format. So, my configuration files will be in YAML.
- I want to make a template which will make project tree based on choice of enabling GAN options. So, I have put a GAN
  option in ```new_project.py``` file. It will give me a bit of freedom while creating project tree as I work with GAN
  most of the time.
- I want to practice more OOP coding. Currently, my coding style isn't OOP that much. I want to become more familiar
  with OOP styles.

## Prerequisites

- NVIDIA GPU + CUDA cuDNN
- Python >= 3.7
- PyTorch >= 1.4

### Optional

- pylint - `conda install pylint` or `pip install pylint` - For formatting according to PEP8 formatting. Not necessary
  if you're not too much into formatting and convention stuff.

## How to run

* **Create a project template with GAN option**

```
    python new_project.py --name MyNewProjectWithGAN --path ~/ --enable_gan yes
```

* **Create a project template without GAN option**

```
    python new_project.py --name MyNewProjectWithoutGAN --path ~/ --enable_gan no
```

* **Optional - After creating project/editing your code, check for code errors, format errors etc. with pylint. Go to
  terminal, cd to your project directory and run the following code.**

```
    pylint ${PWD}
```

## Folder Structure Created for Project

  ```
  PyTorch_Project_Template/
  │
  ├── test_gan.py/test.py - evaluation of trained model
  ├── train_gan.py/train.py - main script to start training
  ├── config_gan.yaml/config.yaml - demo config file
  │
  ├── base/ - abstract base classes
  │   ├── base_data_loader.py - abstract base class for data loaders
  │   ├── base_model.py - abstract base class for models
  │   └── base_trainer_gan.py/base_trainer.py - abstract base class for trainers
  │
  ├── data_loader/ - dataloader and dataset
  │   ├── data_loader.py
  |   └── dataset.py 
  │
  ├── model/ - models, losses, and metrics
  │   ├── layer_utils.py
  │   ├── loss.py
  │   ├── metric.py
  │   └── model.py
  │
  ├── trainer/ - trainers
  │   └── trainer_gan.py/trainer.py
  │
  └── utils/
      ├── logger.py - class for train logging
      ├── util.py
      ├── visualization.py - class for tensorboardX visualization support
      └── ...
  ```

## Example configuration file options in config_gan.yaml

```
name: MyGAN
n_gpu: 1

data_loader:
  type: CustomDataset
  args:
    data_dir: data/
    batch_size: 1
    shuffle: False
    validation_split: 0
    num_workers: 4

generator:
  type: ResNetGenerator
  args:
    input_nc: 3
    output_nc: 3

discriminator:
  type: NLayerDiscriminator
  args:
    input_nc": 3

loss:
  adversarial: wgan_gp_loss
  content: perceptual_loss

metrics: [
    PSNR
]

optimizer:
  type: Adam
  args:
    lr: 0.0001
    betas: [
        0.5,
        0.999
    ]
    weight_decay: 0
    amsgrad: True

lr_scheduler:
  type: LambdaLR
  args:
    lr_lambda: origin_lr_scheduler

trainer:
  epochs: 300
  save_dir: saved/
  save_period: 5
  verbosity: 2
  monitor: max PSNR
  tensorboardX: True
  log_dir: saved/runs

others:
  gp_lambda: 10
  content_loss_lambda: 100

```

## To-do

- [x] Make a basic template/Steal a template ([You](https://github.com/victoresque/pytorch-template) have received 75%
  thanks and [You](https://github.com/fourson/DeblurGAN-pytorch) have received 22% thanks. I will keep 3% as I am still
  editing your codes. Kudos!)
- [x] Make option for GAN/No GAN project
- [x] Check for tree copy errors - Project creation is OK!
- [ ] Make projects with example models
- [ ] Train some small/medium models to check every method/class
- [ ] Make projects with example models with GAN
- [ ] Train some small/medium GAN models to check every method/class
- [ ] *I repeat - I didn't check the codes yet!!!*
- [ ] *Please read the previous comment!*

## Acknowledgements

The organization of this project is based on [PyTorch Template Project](https://github.com/victoresque/pytorch-template)
and [DeblurGAN](https://github.com/fourson/DeblurGAN-pytorch). I have given you guys thanks in the to-do list already.
If I earn lots of money, I will give you more *thanks*. I can't give you money. I am broke as hell. Cheers!