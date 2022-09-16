#!/bin/bash

#cities=( 4611040 3520005 2443027 )
#da_nb=( 1118 3702 249 )

#index=0
#for c in "${cities[@]}"
#do
#  nb=("${da_nb[$index]}")
#	nb=$((nb+200))
#	for d in $(eval echo "{0..$nb..200}")
#	do
#      qsub launch_compute_stats.sh /nobackup/geompr/Canada_new $c $d
#  done
#	index=$((index+1))
#done

da_nb=56590
nb=$((da_nb+2000))
for i in $(eval echo "{0..$nb..2000}")
do
	qsub launch_compute_stats.sh /nobackup/geompr/Canada_new 2443027 $i
done
