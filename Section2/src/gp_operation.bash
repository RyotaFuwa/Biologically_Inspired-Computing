#!/usr/bin/env bash


for i in {11..20}; do
./gp.bash 1|tail -n 1 > data/op/$i.txt
done


