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

# glibc-dns-testing: A glibc-based image for testing DNS resolution behavior.
#
# This image exists to test that glibc-based programs can resolve Service DNS
# names correctly. It complements agnhost (which uses Alpine/musl) to ensure
# Kubernetes DNS works correctly regardless of which C library the application
# uses. See issue #10161 for the original discussion about libc DNS differences.
#
# Key differences between glibc and musl DNS resolution:
# - glibc queries nameservers sequentially; musl queries in parallel
# - glibc and musl handle ndots and search domains differently
# - The hostname command behaves differently between glibc and musl

ARG BASEIMAGE
FROM $BASEIMAGE

CROSS_BUILD_COPY qemu-QEMUARCH-static /usr/bin/

RUN apt-get update && \
    apt-get install -y dnsutils && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ADD https://github.com/coredns/coredns/releases/download/v1.5.0/coredns_1.5.0_linux_BASEARCH.tgz /coredns.tgz
RUN tar -xzvf /coredns.tgz && rm -f /coredns.tgz
