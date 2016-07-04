#!/usr/bin/env bash
# Copyright 2015 The Kubernetes Authors.
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

REGISTRY_HOST=${REGISTRY_HOST:?no host}
REGISTRY_PORT=${REGISTRY_PORT:-5000}
REGISTRY_CA=${REGISTRY_CA:-/var/run/secrets/kubernetes.io/serviceaccount/ca.crt}
FORWARD_PORT=${FORWARD_PORT:-5000}
sed -e "s/%HOST%/$REGISTRY_HOST/g" \
	-e "s/%PORT%/$REGISTRY_PORT/g" \
	-e "s/%FWDPORT%/$FORWARD_PORT/g" \
	-e "s|%CA_FILE%|$REGISTRY_CA|g" \
	</proxy.conf.in >/proxy.conf

# wait for registry to come online
while ! host "$REGISTRY_HOST" &>/dev/null; do
	printf "waiting for %s to come online\n" "$REGISTRY_HOST"
	sleep 1
done

printf "starting proxy\n"
exec haproxy -f /proxy.conf "$@"
