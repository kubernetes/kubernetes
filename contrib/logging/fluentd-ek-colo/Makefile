# Makefile for scenario where the Elasticsearch and Kibana containers
# have to be co-located inside the same pod.

KUBECTL=kubectl.sh

up:	logger-up es-kibana-up

down:	logger-down es-kibana-down


es-kibana-up:	
	-${KUBECTL} create -f es-service.yml
	-${KUBECTL} create -f es-kibana-pod.yml
	-${KUBECTL} create -f kibana-service.yml

es-kibana-down:
	-${KUBECTL} delete pods elasticsearch-kibana-pod
	-${KUBECTL} delete service kibana-colo
	-${KUBECTL} delete service elasticsearch

update:	
	-${KUBECTL} delete pods elasticsearch-kibana-pod
	-${KUBECTL} create -f es-kibana-pod.yml

logger-up:
	-${KUBECTL} create -f synthetic_0_25lps.yml

logger-down:
	-${KUBECTL} delete pods synthetic-logger-0.25lps-pod

get:	
	${KUBECTL} get pods
	${KUBECTL} get services

net:	
	gcutil getforwardingrule elasticsearch
	gcutil getforwardingrule kibana-colo

firewall:	
	gcutil addfirewall --allowed=tcp:5601,tcp:9200,tcp:9300 --target_tags=kubernetes-minion kubernetes-elk-example


rmfirewall:
	gcutil deletefirewall -f kubernetes-elk-example

