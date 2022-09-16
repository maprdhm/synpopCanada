#!/bin/bash

provinces=( 10 11 12 13 24 35 46 47 48 59 60 61 62 )
da_nb=( 1073 295 1658 1454 13658 20160 2183 2474 5803 7617 67 98 50 )
index=0
for i in "${provinces[@]}"
do
	nb=("${da_nb[$index]}")
	nb=$((nb+250))
	for i in $(eval echo "{0..$nb..250}")
	do
		echo $i $j
  		qsub launch_generate_synth_pop.sh /nobackup/geompr $i $j False
	done
	index=$((index+1))
done
