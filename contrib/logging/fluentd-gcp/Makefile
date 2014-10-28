# Makefile for a synthetic logger to be logged
# by GCP. The cluster must have been created with
# the environment variable FLUENTD_GCP set to 'true'.

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

