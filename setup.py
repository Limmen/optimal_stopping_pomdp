from setuptools import setup

setup(name='optimal_stopping_pomdp',
      version='0.0.1',
      install_requires=['numpy', 'pulp', 'stable-baselines3'],
      author='Kim Hammar',
      author_email='hammar.kim@gmail.com',
      description='Basic implementation of an optimal stopping POMDP.',
      license='Creative Commons Attribution-ShareAlike 4.0 International',
      keywords='POMDP Optimal Stopping',
      url='https://github.com/Limmen/optimal_stopping_pomdp',
      download_url='https://github.com/Limmen/optimal_stopping_pomdp',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Developers',
          'Topic :: Software Development :: Build Tools',
          'Programming Language :: Python :: 3.8'
      ]
      )