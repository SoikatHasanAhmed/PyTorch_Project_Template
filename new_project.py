""" ~~~ Made by Sayed Nadim ~~~
This script initializes new pytorch project with the template files.
If you want a template with GAN options, then,
Run `python3 new_project.py --name MyNewProjectWithGAN --path ~/ --enable_gan yes`
and a new project named MyNewProject with GAN options will be made in the given path.
If you don't want any GAN options, then,
Run `python3 new_project.py --name MyNewProjectWithoutGAN --path ~/ --enable_gan no`
and a new project named MyNewProject without GAN options will be made.
"""


import argparse
from pathlib import Path
from shutil import copytree, ignore_patterns

DESCRIPTION = """ ~~~ Made by Sayed Nadim ~~~
This script initializes new pytorch project with the template files.
If you want a template with GAN options, then,
Run `python3 new_project.py --name MyNewProjectWithGAN --path ~/ --enable_gan yes` 
and a new project named MyNewProject with GAN options will be made in the given path.
If you don't want any GAN options, then,
Run `python3 new_project.py --name MyNewProjectWithoutGAN --path ~/ --enable_gan no` 
and a new project named MyNewProject without GAN options will be made.
"""

print("-" * 80)
print(DESCRIPTION)
print("-" * 80)

parser = argparse.ArgumentParser()
parser.add_argument('--name', help='Name for your new project', default='NewProject')
parser.add_argument('--path', help='Path for your new project', default='../')
parser.add_argument('--enable_gan', help='Enable GAN option.', type=str, default='no')
args = vars(parser.parse_args())
parser.print_help()
print('-' * 80)

current_dir = Path()
assert (current_dir / 'new_project.py').is_file(), \
    'Script should be executed in the pytorch-template directory'
assert args['name'] is not None, "Specify a name for the new project"
assert args['path'] is not None, "Specify a path for the new project"
assert args['enable_gan'] is not None, "Specify option for enable/disable GAN for the new project"

# project_name = Path(sys.argv[1])
# project_name = args['name']
target_dir = args['path'] + '/' + args['name']  # current_dir / project_name

gan_option = args['enable_gan']

if gan_option == 'no':
    ignore = [".git", "data", "saved", "new_project.py", "LICENSE", ".flake8", "README.md",
              "__pycache__", "trainer_gan.py", "base_trainer_gan.py", "train_gan.py",
              "test_gan.py", "config_gan.yaml"]
    copytree(current_dir, target_dir, ignore=ignore_patterns(*ignore))
    print('New project initialized at', target_dir, 'with no GAN option.')
elif gan_option == 'yes':
    ignore = [".git", "data", "saved", "new_project.py", "LICENSE", ".flake8", "README.md",
              "__pycache__", "config.yaml", "trainer.py", "base_trainer.py", "train.py", "test.py"]
    copytree(current_dir, target_dir, ignore=ignore_patterns(*ignore))
    print('New project initialized at', target_dir, 'with GAN option.')
else:
    raise ValueError("Please select \'yes\' or \'no\' for enable_gan option.")
