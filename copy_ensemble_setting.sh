#!/bin/bash

cd $(dirname $0)/..

prefix=$1
setting_dir=$2
output_dir=$3
num=$4

mkdir -p $output_dir

passive=`ls $setting_dir/*/passive/${prefix}simulate_setting.json | xargs -IXX sh -c "echo -n 'XX '; cat XX| jq '.score * (.validate_report.gain+.optimize_report.gain)'" | sort -n -k2 | awk '{print $1}' | head -$num`
active=`ls $setting_dir/*/active/${prefix}simulate_setting.json | xargs -IXX sh -c "echo -n 'XX '; cat XX| jq '.validate_report.gain + .optimize_report.gain'" | sort -n -k2 | awk '{print $1}' | tail -$num`

files=`echo $passive $active | xargs -n 1`

index=0
for file in $files; do
  echo $file
  cp $file ${output_dir}/${index}_${prefix}simulate_setting.json
  index=$((index + 1))
done
