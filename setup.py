from setuptools import setup, find_packages

setup(
  name = 'flowmatching-tabular',
  packages = find_packages(exclude=['assets']),
  version = '0.1.0',
  license='MIT',
  description = 'Flow Matching with BDTs',
  long_description_content_type = 'text/markdown',
  author = 'Flore N'kam Suguem',
  author_email = 'flore.n_kam_suguem@math.univ-toulouse.fr',
  url = 'https://https://github.com/FloAI/flowmatching_tabular',
  keywords = [
    'artificial intelligence',
    'flow matching',
    'regression',
  ],
  install_requires=[
    'xgboost>=2.0.0',
    'scikit-learn>=1.3',
    'tqdm>=4.6',
    'tqdm_joblib>=0.0.3',
  ],
  setup_requires=[
    'pytest-runner',
  ],
  tests_require=[
    "joblib==1.3.0",
    "scikit-learn==1.3.2",
    "tqdm==4.66.3",
    "tqdm_joblib==0.0.3",
    "xgboost==2.0.0"
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.12',
  ],
)
