#!/bin/bash

a=$1
b="${a%.*}.log"
nohup $1 > ./tmp/$b 2>&1 &
b="nohup_pid.txt"
echo $! > ./tmp/$b