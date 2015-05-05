#!/bin/sh
# Copyright 2014 Google Inc. All rights reserved.
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

#
# This script builds a configuration for Prometheus based on command line
# arguments and environment variables and starts the Prometheus server.
#
# Sample usage (to be run inside Kubernetes-created docker container).
#  ./run_prometheus -t KUBERNETES_RO -d /tmp/prometheus
#

show_usage() {
  echo "usage: ./run_prometheus -t TARGET_1,TARGET_2 -d data_directory"
  echo "where"
  echo " -t List of services to be monitored. Each service T should be described by"
  echo "    the T_SERVICE_HOST and T_SERVICE_PORT env variables."
  echo " -d Location where the config file and metrics will be written."
}

build_config() {
  echo >$1 'global: { scrape_interval: "10s" evaluation_interval: "10s"}'
  local target
  for target in ${2//,/ }; do
    local host_variable=$target"_SERVICE_HOST"
    local port_variable=$target"_SERVICE_PORT"
    local host=`eval echo '$'$host_variable`
    local port=`eval echo '$'$port_variable`
    echo "Checking $target"
    if [ -z $host ]; then
      echo "No env variable for $host_variable."
      exit 3
    fi
    if [ -z $port ]; then
      echo "No env variable for $port_variable."
      exit 3
    fi
    local target_address="http://"$host":"$port"/metrics"
    echo >>$1 "job: { name: \"${target}\" target_group: { target: \"${target_address}\" } }"
  done
}

while getopts :t:d: flag; do
  case $flag in
  t) # targets.
    targets=$OPTARG
    ;;
  d) # data location
    location=$OPTARG
    ;;
  \?)
    echo "Unknown parameter: $flag"
    show_usage
    exit 2
    ;;
  esac
done

if [ -z $targets ] || [ -z $location ]; then
  echo "Missing parameters."
  show_usage
  exit 2
fi

mkdir -p $location
config="$location/config.pb"
storage="$location/storage"

echo "-------------------"
echo "Starting Prometheus with:"
echo "targets: $targets"
echo "config: $config"
echo "storage: $storage"

build_config $config $targets
echo "-------------------"
echo "config file:"
cat $config
echo "-------------------"

exec /bin/prometheus \
  "-logtostderr" \
  "-config.file=$config" \
  "-storage.local.path=$storage" \
  "-web.console.libraries=/go/src/github.com/prometheus/prometheus/console_libraries" \
  "-web.console.templates=/go/src/github.com/prometheus/prometheus/consoles"
