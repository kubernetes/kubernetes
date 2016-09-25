#! /bin/bash

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

# This volume is assumed to exist and is shared with parent of the init
# container. It contains the redis installation.
INSTALL_VOLUME="/opt"

# This volume is assumed to exist and is shared with the peer-finder
# init container. It contains on-start/change configuration scripts.
WORK_DIR="/work-dir"

VERSION="3.2.0"

for i in "$@"
do
case $i in
    -i=*|--install-into=*)
    INSTALL_VOLUME="${i#*=}"
    shift
    ;;
    -w=*|--work-dir=*)
    WORK_DIR="${i#*=}"
    shift
    ;;
    *)
    # unknown option
    ;;
esac
done

echo installing config scripts into "${WORK_DIR}"
mkdir -p "${WORK_DIR}"
cp /on-start.sh "${WORK_DIR}"/
cp /peer-finder "${WORK_DIR}"/

echo installing redis-"${VERSION}" into "${INSTALL_VOLUME}"
mkdir -p "${INSTALL_VOLUME}"
mv /redis "${INSTALL_VOLUME}"/redis
