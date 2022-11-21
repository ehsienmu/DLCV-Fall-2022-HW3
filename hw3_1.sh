#!/bin/bash

# TODO - run your inference Python3 code
python3 part1_test.py --input_images_dir $1 --input_json_file $2 --output_file $3
# python3 part1_test.py --input_images_dir ./hw3_data/p1_data/val/ --input_json_file ./hw3_data/p1_data/id2label.json --output_file ./part1_output.csv