#!/usr/bin/env bash

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

DEB_ARCH=$(dpkg --print-architecture)

# http://security.debian.org/debian-security/dists/jessie/updates/InRelease is missing
# entries for some platforms, so we just remove the last line in sources.list in
# /etc/apt/sources.list which is "deb http://deb.debian.org/debian jessie-updates main"

case ${DEB_ARCH} in
    s390x|arm64|ppc64el)
        # we have to use the archive mirrors.
        # See: https://lists.debian.org/debian-devel-announce/2019/03/msg00006.html
        echo "deb http://archive.debian.org/debian/ jessie main contrib non-free" | tee /etc/apt/sources.list
        echo "deb-src http://archive.debian.org/debian/ jessie main contrib non-free" | tee -a /etc/apt/sources.list
       ;;
esac

exit 0
