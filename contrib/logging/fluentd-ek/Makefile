# Makefile for Fluentd to Elastiscsearch and Kibana configured
# in separate pods.


.PHONY: up dow es-up kibana-up es-down kibana-down update \
        logger-up logger-down get net firewall rmfirewall

KUBECTL=kubectl.sh

up:	logger-up es-up kibana-up

down:	logger-down es-down kibana-down


es-up:	
	-${KUBECTL} create -f es-pod.yml
	-${KUBECTL} create -f es-service.yml

kibana-up:
	-${KUBECTL} create -f kibana-pod.yml
	-${KUBECTL} create -f kibana-service.yml

es-down:
	-${KUBECTL} delete pods elasticsearch-pod
	-${KUBECTL} delete service elasticsearch

kibana-down:
	-${KUBECTL} delete pods kibana-pod
	-${KUBECTL} delete service kibana


update:	
	-${KUBECTL} delete pods kibana-pod
	-${KUBECTL} create -f kibana-pod.yml

logger-up:
	-${KUBECTL} create -f synthetic_0_25lps.yml

logger-down:
	-${KUBECTL} delete pods synthetic-logger-0.25lps-pod

get:	
	${KUBECTL} get pods
	${KUBECTL} get services

net:	
	gcutil getforwardingrule elasticsearch
	gcutil getforwardingrule kibana

firewall:	
	gcutil addfirewall --allowed=tcp:5601,tcp:9200,tcp:9300 --target_tags=kubernetes-minion kubernetes-elk-example
 

rmfirewall:
	gcutil deletefirewall -f kubernetes-elk-example

