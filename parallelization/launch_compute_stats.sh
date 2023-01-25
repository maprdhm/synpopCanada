#!/bin/bash

# This script must be run with qsub

#$ -cwd -V
#$ -l h_rt=24:00:00
#$ -l h_vmem=20G
#$ -m e
#$ -M m.predhumeau@leeds.ac.uk
#$ -o ./logs
#$ -e ./errors

if [ "$#" != "4" ]; then
  echo "usage: qsub $0 <path> <city-code> <from-DA-index> <scenario>"
  exit 1
fi

python compute_stats.py $1 $2 $3 $4
