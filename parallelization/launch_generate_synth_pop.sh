#!/bin/bash

# This script must be run with qsub

#$ -cwd -V
#$ -l h_rt=48:00:00
#$ -l h_vmem=20G
#$ -o ./logs
#$ -e ./logs

if [ "$#" != "4" ]; then
  echo "usage: qsub $0 <path-to-files> <province-code> <from-DA-index> <fast>"
  exit 1
fi

python generate_synth_pop.py $1 $2 $3 $4
