#!/bin/bash

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..
exclude_dir="${KUBE_ROOT}/Godeps"
exclude_file1="datafile.go"
exclude_file2="${KUBE_ROOT}/contrib/mesos/pkg/scheduler/doc.go"

find $KUBE_ROOT -path $exclude_dir -prune -o -name '*.go' ! -name $exclude_file1 ! -wholename $exclude_file2 | while read filename
do
	#sed -i 's/\/\//\/\/\ /g;/\/\//s/ \+/ /;s/:\/\/ /:\/\//' "${filename}"
	sed -i 's/\/\//\/\/\ /g;s/\/\/ \+/\/\/ /;s/\:\/\/ /\:\/\//g;s/\/\/ $/\/\//' "${filename}"
	#echo $filename
done
