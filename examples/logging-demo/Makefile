# Makefile for launching syntheitc logging sources (any platform)
# and for reporting the forwarding rules for the
# Elasticsearch and Kibana pods for the GCE platform.


.PHONY: up down logger-up logger-down logger10-up logger10-downget net

KUBECTL=../../cluster/kubectl.sh

up:	logger-up logger10-up

down:	logger-down logger10-down


logger-up:
	-${KUBECTL} create -f synthetic_0_25lps.yaml --namespace=kube-system

logger-down:
	-${KUBECTL} delete pods synthetic-logger-0.25lps-pod --namespace=kube-system

logger10-up:
	-${KUBECTL} create -f synthetic_10lps.yaml --namespace=kube-system

logger10-down:
	-${KUBECTL} delete pods synthetic-logger-10lps-pod --namespace=kube-system

get:	
	${KUBECTL} get pods --namespace=kube-system
	${KUBECTL} get replicationControllers --namespace=kube-system
	${KUBECTL} get services --namespace=kube-system

net:	
	${KUBECTL} get services elasticsearch-logging --namespace=kube-system -o json
	${KUBECTL} get services kibana-logging --namespace=kube-system -o json
