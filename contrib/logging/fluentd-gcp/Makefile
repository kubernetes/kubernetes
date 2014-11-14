# Makefile for a synthetic logger to be logged
# by GCP. The cluster must have been created with
# the variable LOGGING_DESTINATION=GCP.

.PHONY: up down logger-up logger-down get

KUBECTL=kubectl.sh

up:	logger-up

down:	logger-down


logger-up:
	-${KUBECTL} create -f synthetic_0_25lps.yml

logger-down:
	-${KUBECTL} delete pods synthetic-logger-0.25lps-pod

get:	
	${KUBECTL} get pods

