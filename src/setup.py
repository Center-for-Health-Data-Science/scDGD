from setuptools import setup, find_packages
#import
setup(name='scDGD',
      version="0.2",
      #description='Classes for internal representations',
      author='Viktoria Schuster',
      #url='https://github.com/viktoriaschuster/DGD_paper_experiments',
      packages=['scDGD','scDGD.classes','scDGD.models','scDGD.functions']
      #packages=find_packages()#,
      #package_dir = {'dgdExp': 'src'}
     )

# install by going to directory and calling 
# python3 setup.py build
# python3 setup.py install --user
# --user needs to be used on this computer because I do not have sudo rights here