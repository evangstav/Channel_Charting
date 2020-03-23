# Channel Charting

This repository contains several implementations of Deep Indoor Channel Charting techniques, developed during a personal research project aa part of my Master in Communications and Systems Engineering at Denmark's Technical University (DTU).

## Goal 
Explore the application of Deep Learning in Indoor Channel Charting. Initially I we implemented supervised and unsupervised algorithm to perform Channel Charting. We also explored some simple semi supervised algorithms, that make use of easily attainable unlabeled data.

## Implementations
We include the implementations of:
1) Supervised Classifier for predicting in which slice of the given space a transmitter resides.
2) A Supervised Regressor for predicting the exact location of the trasmitter
3) An Unsupervised Autoencoder to learn a low dimensional embedding to be used as a "map".
3) A SemiSupervised Classifier performing almost as well as 2) with 10% of labeled data.

## In Progress
Try to learn the low dimensional mapping using two losses and mixing 1) and 3).


## Copyrights
Copyright (c) 2020 Evangelos Stavropoulos

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.