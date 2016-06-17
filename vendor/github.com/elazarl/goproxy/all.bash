#!/bin/bash

go test || exit
for action in $@; do go $action; done

mkdir -p bin
find regretable examples/* ext/* -maxdepth 0 -type d | while read d; do
	(cd $d
	go build -o ../../bin/$(basename $d)
	find *_test.go -maxdepth 0 2>/dev/null|while read f;do
		for action in $@; do go $action; done
		go test
		break
	done)
done
