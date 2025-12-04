#!/bin/bash
export PYTHONPATH=$PYTHONPATH:.
.venv/bin/python markov/experiment.py
.venv/bin/python markov/plots.py