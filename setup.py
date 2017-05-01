
# coding: utf-8

# In[1]:

from setuptools import setup

setup(name='fastica_lz',
      version='0.1',
      description='implementation and optimization of FastICA algorithm in python ',
      url='https://github.com/663project/fastica_lz.git',
      author='blaire&liver',
      author_email='ml390@duke.edu',
      license='MIT',
      packages=['fastica_lz'],
      install_requires=[
        'scipy', 'numexpr','numpy',
      ],
      zip_safe=False)

