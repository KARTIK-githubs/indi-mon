> **Run a prediction.** Use Python 3.11, install `python -m pip install -r requirements.txt`, then run `python test_pkl_model.py /path/to/photo.jpg` from the repository root. The command loads the model bundled with the repository. Use the bundled pickle from a trusted checkout. Low confidence indicates an unknown result.

# INDIMON

A team project for classifying images of Indian monuments. The repository includes TensorFlow training and inference scripts, saved model artifacts and a static web demo.

## Repository layout

- [train_monument_model.py](train_monument_model.py) trains the classifier on a local dataset.
- [build_negative_dataset.py](build_negative_dataset.py) prepares negative examples for the dataset.
- [test.py](test.py) and [test_pkl_model.py](test_pkl_model.py) are local inference scripts. Review local paths before running them.
- [index.html](index.html) contains the demo interface.

## Attribution

This repository is a fork of [Sandraa16012007/indi-mon](https://github.com/Sandraa16012007/indi-mon). See the upstream repository and Git history for original project attribution.
