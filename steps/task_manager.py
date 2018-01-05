import os
import subprocess
import sys
from importlib import import_module


class Task():
    def __init__(self, trainer):
        self.trainer = trainer

    def substitute(self, user_solution, user_config):
        self.modify_config(user_config)
        self.modify_pipeline(user_solution, user_config)
        self.modify_trainer(user_solution, user_config)
        return self.trainer

    def modify_trainer(self, user_solution, user_config):
        pass

    def modify_config(self, user_config):
        pass

    def modify_pipeline(self, user_solution, user_config):
        pass


class TaskSolutionParser(object):
    """Todo:
    exit doesn't work on exceptions and leaves converted .py file out there
    """

    def __init__(self, filepath):
        self.filepath = filepath

    def __enter__(self):
        if self.filepath.endswith('.ipynb'):
            cmd = 'jupyter nbconvert --to python {}'.format(self.filepath)
            subprocess.call(cmd, shell=True)
            filepath = self.filepath.replace('.ipynb', '.py')
        else:
            filepath = self.filepath

        module_dir, module_filename = os.path.split(filepath)
        module_name = module_filename.replace('.py', '')
        sys.path.append(module_dir)
        task_solution = vars(import_module(module_name))
        return task_solution

    def __exit__(self, exc_type, exc_value, traceback):
        if self.filepath.endswith('.ipynb'):
            filepath = self.filepath.replace('.ipynb', '.py')
            cmd = 'rm {}'.format(filepath)
            subprocess.call(cmd, shell=True)
