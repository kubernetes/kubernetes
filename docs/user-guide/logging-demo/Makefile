# Makefile for launching synthetic logging sources (any platform)
# and for reporting the forwarding rules for the
# Elasticsearch and Kibana pods for the GCE platform.
# For examples of how to observe the ingested logs please
# see the appropriate getting started guide e.g.
# Google Cloud Logging: https://github.com/GoogleCloudPlatform/kubernetes/blob/master/docs/getting-started-guides/logging.md
# With Elasticsearch and Kibana logging: https://github.com/GoogleCloudPlatform/kubernetes/blob/master/docs/getting-started-guides/logging-elasticsearch.md

.PHONY: up down logger-up logger-down logger10-up logger10-down

up:	logger-up logger10-up

down:	logger-down logger10-down

logger-up:
	kubectl create -f synthetic_0_25lps.yaml

logger-down:
	kubectl delete pod synthetic-logger-0.25lps-pod

logger10-up:
	kubectl create -f synthetic_10lps.yaml

logger10-down:
	kubectl delete pod synthetic-logger-10lps-pod

