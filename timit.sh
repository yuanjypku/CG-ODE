#!/bin/bash
echo "Begin to test time !"

# 无heavyBall
now=`date +'%Y-%m-%d %H:%M:%S'`
start_time1=$(date --date="$now" +%s);
 
python run_models_covid.py

now=`date +'%Y-%m-%d %H:%M:%S'`
end_time1=$(date --date="$now" +%s);
time1=$((end_time1-start_time1))"s"

# 有heavyBall
now=`date +'%Y-%m-%d %H:%M:%S'`
start_time2=$(date --date="$now" +%s);
 
python run_models_covid.py --heavyBall

now=`date +'%Y-%m-%d %H:%M:%S'`
end_time2=$(date --date="$now" +%s);
time2=$((end_time2-start_time2))"s"

echo "NODE:"
echo $time1
echo "HBODE:"
echo $time2

