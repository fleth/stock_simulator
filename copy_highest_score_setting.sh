#!/bin/bash

cd $(dirname $0)/..

prefix=$1
setting_dir=$2

highest=`ls ${setting_dir}/${prefix}simulate_setting.json ${setting_dir}/tmp/*_${prefix}simulate_setting.json | xargs -IXX sh -c "echo -n 'XX ';cat XX | jq '.score'" | awk '{print $2, $1}' | sort -n | awk '{print $2}' | head -1`

cp $highest ${setting_dir}/${prefix}simulate_setting.json

passive=`ls ${setting_dir}/passive/${prefix}simulate_setting.json ${setting_dir}/${prefix}simulate_setting.json ${setting_dir}/tmp/*_${prefix}simulate_setting.json | xargs -IXX sh -c "echo -n 'XX ';cat XX | jq '.score'" | awk '{print $2, $1}' | sort -n | awk '{print $2}' | head -1`

cp $passive ${setting_dir}/passive/${prefix}simulate_setting.json

active=`ls ${setting_dir}/active/${prefix}simulate_setting.json ${setting_dir}/${prefix}simulate_setting.json ${setting_dir}/tmp/*_${prefix}simulate_setting.json | xargs -IXX sh -c "echo -n 'XX ';cat XX | jq -c '[.optimize_report.gain, .validate_report.gain]' | sed 's/\[//g;s/\]//g;s/,/ /g'" | awk '{OFMT="%.0f"} {print $2+$3, $1}' | sort -n | awk '{print $2}' | tail -1`

cp $active ${setting_dir}/active/${prefix}simulate_setting.json
