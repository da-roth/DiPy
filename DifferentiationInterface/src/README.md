# Installation instruction:

# Package local setup
pip install -e . 

# Install twine
pip install twine

# Create wheel from package
python setup.py sdist bdist_wheel

# Go into dist folder
cd dist

# Upload to PyPi
python -m twine upload *





###
### Package versions currently from colab (their might be issues when using the newest due to cross compatibility)
###
TensorFlow version: 2.15.0
PyTorch version: 2.3.0+cu121
JAX version: 0.4.26
JAX lib version: 0.4.26
ml-dtypes version: 0.2.0