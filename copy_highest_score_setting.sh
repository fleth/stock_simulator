#!/bin/bash

cd $(dirname $0)/..

prefix=$1

highest=`ls settings/${prefix}simulate_setting.json settings/tmp/*_${prefix}simulate_setting.json | xargs -IXX sh -c "echo -n 'XX ';cat XX | jq '.validate_score'" | awk '{print $2, $1}' | sort -n | awk '{print $2}' | head -1`

cp $highest settings/${prefix}simulate_setting.json

