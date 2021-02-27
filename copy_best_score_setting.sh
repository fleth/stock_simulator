#!/bin/bash

cd $(dirname $0)/..

prefix=$1
setting_dir=$2
output_dir=$3

mkdir -p $output_dir

active=`find $setting_dir -name "${prefix}simulate_setting.json" | xargs -IXX sh -c "echo -n 'XX '; cat XX | jq '.validate_report.gain + .optimize_report.gain'" | awk '{print $1}' | sort -n | tail -n1`
passive=`find $setting_dir -name "${prefix}simulate_setting.json" | xargs -IXX sh -c "echo -n 'XX '; cat XX | jq '.score'" | awk '{print $1}' | sort -nr | tail -n1`

short_active=`find $setting_dir -name "${prefix}short_simulate_setting.json" | xargs -IXX sh -c "echo -n 'XX '; cat XX | jq '.validate_report.gain + .optimize_report.gain'" | awk '{print $1}' | sort -n | tail -n1`
short_passive=`find $setting_dir -name "${prefix}short_simulate_setting.json" | xargs -IXX sh -c "echo -n 'XX '; cat XX | jq '.score'" | awk '{print $1}' | sort -nr | tail -n1`

files=`echo $active $passive | xargs -n 1`

index=`ls $output_dir | wc -l`
for file in $files; do
  echo $file
  cp $file ${output_dir}/${index}_${prefix}simulate_setting.json
  index=$((index + 1))
done

files=`echo $short_active $short_passive | xargs -n 1`

index=`ls $output_dir | wc -l`
for file in $files; do
  echo $file
  cp $file ${output_dir}/${index}_${prefix}short_simulate_setting.json
  index=$((index + 1))
done


