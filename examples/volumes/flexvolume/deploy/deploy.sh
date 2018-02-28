#!/bin/sh

# Copyright 2018 The Kubernetes Authors.
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

set -o errexit
set -o pipefail

# TODO change to your desired driver.
VENDOR=${VENDOR:-k8s}
DRIVER=${DRIVER:-dummy}

# Assuming the single driver file is located at /$DRIVER inside the DaemonSet image.

driver_dir=$VENDOR${VENDOR:+"~"}${DRIVER}
if [ ! -d "/flexmnt/$driver_dir" ]; then
  mkdir "/flexmnt/$driver_dir"
fi

tmp_driver=.tmp_$DRIVER
cp "/$DRIVER" "/flexmnt/$driver_dir/$tmp_driver"
mv -f "/flexmnt/$driver_dir/$tmp_driver" "/flexmnt/$driver_dir/$DRIVER"

while : ; do
  sleep 3600
done
