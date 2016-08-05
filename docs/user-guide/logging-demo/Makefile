# Copyright 2016 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Makefile for launching synthetic logging sources (any platform)
# and for reporting the forwarding rules for the
# Elasticsearch and Kibana pods for the GCE platform.
# For examples of how to observe the ingested logs please
# see the appropriate getting started guide e.g.
# Google Cloud Logging: https://github.com/kubernetes/kubernetes/blob/master/docs/getting-started-guides/logging.md
# With Elasticsearch and Kibana logging: https://github.com/kubernetes/kubernetes/blob/master/docs/getting-started-guides/logging-elasticsearch.md

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

