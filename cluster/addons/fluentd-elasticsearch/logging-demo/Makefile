# Makefile for launching syntheitc logging sources (any platform)
# and for reporting the forwarding rules for the
# Elasticsearch and Kibana pods for the GCE platform.


.PHONY: up down logger-up logger-down logger10-up logger10-downget net

LMKTFYCTL=../../../lmktfyctl.sh

up:	logger-up logger10-up

down:	logger-down logger10-down


logger-up:
	-${LMKTFYCTL} create -f synthetic_0_25lps.yaml

logger-down:
	-${LMKTFYCTL} delete pods synthetic-logger-0.25lps-pod

logger10-up:
	-${LMKTFYCTL} create -f synthetic_10lps.yaml

logger10-down:
	-${LMKTFYCTL} delete pods synthetic-logger-10lps-pod

get:	
	${LMKTFYCTL} get pods
	${LMKTFYCTL} get replicationControllers
	${LMKTFYCTL} get services

net:	
	${LMKTFYCTL} get services elasticsearch-logging -o json
	${LMKTFYCTL} get services kibana-logging -o json
