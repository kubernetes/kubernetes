# Makefile for launching syntheitc logging sources (any platform)
# and for reporting the forwarding rules for the
# Elasticsearch and Kibana pods for the GCE platform.


.PHONY: up down logger-up logger-down logger10-up logger10-downget net

KUBECTL=../../cluster/kubectl.sh

up:	logger-up logger10-up

down:	logger-down logger10-down


logger-up:
	-${KUBECTL} create -f synthetic_0_25lps.yaml

logger-down:
	-${KUBECTL} delete pods synthetic-logger-0.25lps-pod

logger10-up:
	-${KUBECTL} create -f synthetic_10lps.yaml

logger10-down:
	-${KUBECTL} delete pods synthetic-logger-10lps-pod

get:	
	${KUBECTL} get pods
	${KUBECTL} get replicationControllers
	${KUBECTL} get services

net:	
	${KUBECTL} get services elasticsearch-logging -o json
	${KUBECTL} get services kibana-logging -o json
