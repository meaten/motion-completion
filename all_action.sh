#!/bin/bash

for ACTION in "walking" "eating" "smoking" "discussion" "directions" "greeting"  "phoning"  "posing"  "purchases"  "sitting" "sittingdown"  "takingphoto"  "waiting"  "walkingdog" "walkingtogether" "all"
do
    python src/translate.py --residual_velocities --learning_rate 0.005 --action $ACTION --seq_length_out 25 --iteration 10000 --gpu_assignment 1
done
