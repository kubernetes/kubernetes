# Copyright 2017 The Kubernetes Authors.
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

FROM 	quay.io/prometheus/busybox:latest
LABEL	maintainer	"Shyam JVS <shyamjvs@google.com>"

COPY	./bin/etcdctl				/bin/etcdctl
COPY	./bin/prom2json				/bin/prom2json
COPY	./bin/etcd-version-monitor	/bin/etcd-version-monitor

EXPOSE	9101
