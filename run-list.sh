#!/bin/bash

# This runs NewTR with a list of CUTEst problems

[ $# -lt 1 ] && echo "ERROR: Need list of problem" && exit 1
[ ! -f $1 ] && echo "$1 is not a file" && exit 1

dir=cutest-$(date +"%Y-%m-%d-%H:%M")
mkdir -p $dir
cp $1 $dir

k=1
T=$(wc -l $1 | cut -f1 -d" ")

for prob in $(cat $1)
do
  printf "Running problem %4d of %4d: $prob\n" $k $T
  make cutest PROBLEM=$prob > $dir/$prob.out
  k=$((k+1))
done
