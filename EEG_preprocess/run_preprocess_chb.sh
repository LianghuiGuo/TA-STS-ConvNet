#!/bin/bash

for id in 1 2 3 5 6 7 8 9 10 11 13 14 16 17 18 19 20 21 22 23
do
    python chb_preprocess.py --patient_id=$id
done