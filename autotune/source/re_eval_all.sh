#!/bin/bash

rm ../final_result/*.csv

rm ../final_result/*center.pk

python feature_extraction_classifier.py

python match.py

python evaluate_by_global.py


