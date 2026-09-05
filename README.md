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


## Historical experiment environments

The `notebooks/wandb/run-*` and `src/wandb/run-*` directories contain archived
W&B experiment records from 2020. Each `environment-snapshot.md` preserves the
original generated `requirements.txt` package list as documentation, not as an
installation manifest. The training scripts and notebook code do not reference
these snapshots; this repository has no maintained installation manifest or CI
installation workflow.

All 47 snapshots record `tornado==6.0.3`. This version is affected by
[CVE-2026-82397](https://github.com/advisories/GHSA-mpf4-983q-p7j4), a denial of
service in form-encoded request parsing, fixed in
[Tornado 6.5.8](https://www.tornadoweb.org/en/stable/releases/v6.5.8.html).
Archiving the lists preserves experiment provenance without presenting old
package pins as current dependencies. It does not patch an existing environment.

Do not install these historical package lists. To run the experiments again,
create and validate a fresh environment with supported Python and dependencies,
including Tornado 6.5.8 or later if Tornado is needed. The three flagged records used
Python 3.8.1 and old Jupyter/W&B packages; compatibility with a modern environment
has not been tested. Existing vulnerable notebook or other Tornado servers remain
at risk if they accept attacker-controlled requests. Other historical packages
have not been audited. Older Git revisions still contain the original manifests.

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