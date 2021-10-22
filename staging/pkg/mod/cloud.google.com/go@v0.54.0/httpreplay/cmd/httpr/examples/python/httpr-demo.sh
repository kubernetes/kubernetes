#!/bin/sh -e
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This script and the accompanying program, httpr-demo.py, demonstrate how to use
# httpr with Python.
#
# Prerequisites:
# - httpr is on your path.
# - The google-cloud and google-auth Python packages have been installed:
#       pip install --upgrade google-cloud
#       pip install --upgrade google-auth

# Execution:
# 1. Pick a project and a GCS bucket.
# 2. Invoke this script to record an interaction:
#       http-demo.sh PROJECT BUCKET record
# 3. Invoke the script again to replay:
#       http-demo.sh PROJECT BUCKET replay

project=$1
bucket=$2
mode=$3

if [[ $mode != "record" && $mode != "replay" ]]; then
  echo >&2 "usage: $0 PROJECT BUCKET record|replay"
  exit 1
fi

if [[ $(which httpr) == "" ]]; then
  echo >&2 "httpr is not on PATH"
  exit 1
fi

# Start the proxy and wait for it to come up.
httpr -$mode /tmp/demo.replay &
proxy_pid=$!

# Stop the proxy on exit.
# When the proxy is recording, this will cause it to write the replay file.
trap "kill -2 $proxy_pid" EXIT

sleep 1

# Download the CA certificate from the proxy's control port
# and inform Python of the cert via an environment variable.
cert_file=/tmp/httpr.cer
curl -s localhost:8181/authority.cer > $cert_file
export REQUESTS_CA_BUNDLE=$cert_file

# Tell Python to use the proxy.
# If you passed the -port argument to httpr, use that port here.
export HTTPS_PROXY=localhost:8080

# Run the program.
python httpr-demo.py $project $bucket $mode

