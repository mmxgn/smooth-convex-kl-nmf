import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "smooth-convex-kl-nmf",
    version = "0.0.1",
    author = "Emmanouil Theofanis Chourdakis",
    author_email = "e.t.chourdakis@qmul.ac.uk",
    description = ("An implementation of Essid, S. et al Smooth+Convex KL-NMF method for speaker diarization, in python."),
    license = "BSD",
    keywords = "nmf matrix-factorization nonnegative-matrix-factorization",
    url = "https://github.com/mmxgn/smooth-convex-kl-nmf",
    packages=['scnmf', 'tests'],
    long_description=read('README.md'),
    classifiers=[ 'Intended Audience :: Science/Research',
                  'License :: OSI Approved :: BSD License',
                  'Topic :: Multimedia :: Sound/Audio :: Analysis'
    ],
)