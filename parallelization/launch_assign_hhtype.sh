#!/bin/bash

# This script must be run with qsub

#$ -cwd -V
#$ -l h_rt=24:00:00
#$ -l h_vmem=10G
#$ -m e
#$ -M m.predhumeau@leeds.ac.uk
#$ -o ./logs
#$ -e ./errors

if [ "$#" != "4" ]; then
  echo "usage: qsub $0 <path> <province-code> <from-DA-index> <year>"
  exit 1
fi

python assign_hhtype.py $1 $2 $3 $4

