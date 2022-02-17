#!/bin/bash
input=$1
output=$2
cut -f1,3,4,5,9 $input > $output
echo "Sending output to $output"
