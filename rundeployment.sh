#!/bin/bash

CNT=30
for ((i=0; i < CNT; i++)); do
     kubectl create deployment  simple-app$i  --image=rtsp/lighttpd:1.4.73  --replicas=1
done
