#!/bin/sh

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

# View all commands in the output

set -x

# Install build tools

apt-get update
apt-get install -y bash curl gcc make autoconf intltool gperf libxslt-dev docbook-xsl pkg-config xsltproc

# Install dependencies

apt-get install -y libssl-dev libtool libgcrypt-dev liblz4-dev libcap-dev libmount-dev

# Build jemalloc

curl -LO https://github.com/jemalloc/jemalloc/archive/4.3.1.tar.gz
tar -xf 4.3.1.tar.gz
cd jemalloc-4.3.1

JEMALLOC_OTPUT_DIR=${BUILD_DIR}/jemalloc
mkdir -p ${JEMALLOC_OTPUT_DIR}
./autogen.sh --prefix=${JEMALLOC_OTPUT_DIR}
make
make install_lib_shared

cd ..
rm -rf 4.3.1.tar.gz jemalloc-4.3.1

# Build systemd

curl -LO https://github.com/systemd/systemd/archive/v222.tar.gz
tar -xf v222.tar.gz
cd systemd-222

SYSTEMD_BUILD_DIR=${BUILD_DIR}/systemd
./autogen.sh
./configure CFLAGS='-g -O0 -ftrapv' --enable-compat-libs --enable-kdbus --enable-lz4 --prefix=${SYSTEMD_BUILD_DIR}
make
make install

cd ..
rm -rf v222.tar.gz systemd-222

# Build ruby

curl -LO https://cache.ruby-lang.org/pub/ruby/2.3/ruby-2.3.3.tar.gz
tar -xf ruby-2.3.3.tar.gz
cd ruby-2.3.3

RUBY_BUILD_DIR=${BUILD_DIR}/ruby
mkdir -p ${RUBY_BUILD_DIR}
./configure --disable-install-doc --with-jemalloc --with-opt-dir=${JEMALLOC_OTPUT_DIR} --prefix=${RUBY_BUILD_DIR}
make
make install

cd ..
rm -rf ruby-2.3.3.tar.gz ruby-2.3.3

# Build required gems

RUBY_BIN_DIR=${RUBY_BUILD_DIR}/bin
${RUBY_BIN_DIR}/gem install --no-document --bindir=${RUBY_BIN_DIR} fluentd -v 0.12.29
${RUBY_BIN_DIR}/gem install --no-document --bindir=${RUBY_BIN_DIR} fluent-plugin-record-reformer -v 0.8.2
${RUBY_BIN_DIR}/gem install --no-document --bindir=${RUBY_BIN_DIR} fluent-plugin-systemd -v 0.0.5
${RUBY_BIN_DIR}/gem install --no-document --bindir=${RUBY_BIN_DIR} fluent-plugin-google-cloud -v 0.5.2

# Copy dependencies

mkdir ${BUILD_DIR}/lib/
cp /lib/x86_64-linux-gnu/libcrypto.so* ${BUILD_DIR}/lib/
cp /lib/x86_64-linux-gnu/libssl.so* ${BUILD_DIR}/lib/

# Copy build artifacts to the destination directory

cp -R ${BUILD_DIR}/* ${OUTPUT_DIR}/