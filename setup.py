from setuptools import setup, find_packages

requires_list = ["mushroom_rl>=1.4"]

setup(name='mushroom_rl_imitation',
      version='0.1',
      description='Deep Imitation learning based on mushroom_rl library.',
      license='MIT',
      author="Manuel Palermo",
      packages=[package for package in find_packages()
                if package.startswith('mushroom_rl')],
      install_requires=requires_list,
      zip_safe=False,
      )