#!/bin/bash

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

# A script encapsulating a common Dockerimage pattern for installing packages
# and then cleaning up the unnecessary install artifacts.
# e.g. clean-install iptables ebtables conntrack

set -o errexit

# 1. Install & configure dependencies.
# 2. Install fluentd via ruby.
# 3. Remove build dependencies.
# 4. Cleanup leftover caches & files.
BUILD_DEPS="make gcc g++ libc6-dev ruby-dev libffi-dev"

# apt install
apt-get update
echo "${BUILD_DEPS} ca-certificates libjemalloc2 ruby" | xargs apt-get install -y --no-install-recommends

# ruby install
echo 'gem: --no-document' >> /etc/gemrc 
gem install --file Gemfile

# cleanup
echo "${BUILD_DEPS}" | xargs apt-get purge -y --auto-remove -o APT::AutoRemove::RecommendsImportant=false
apt-get clean -y
rm -rf \
   /var/cache/debconf/* \
   /var/lib/apt/lists/* \
   /var/log/* \
   /var/tmp/*

# Ensure fluent has enough file descriptors
ulimit -n 65536
