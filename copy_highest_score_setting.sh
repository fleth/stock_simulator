#!/bin/bash

cd $(dirname $0)/..

prefix=$1
setting_dir=$2

highest=`ls ${setting_dir}/${prefix}simulate_setting.json ${setting_dir}/tmp/*_${prefix}simulate_setting.json | xargs -IXX sh -c "echo -n 'XX ';cat XX | jq '.score'" | awk '{print $2, $1}' | sort -n | awk '{print $2}' | head -1`

cp $highest ${setting_dir}/${prefix}simulate_setting.json

