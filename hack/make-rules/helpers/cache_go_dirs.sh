#!/usr/bin/env bash

# Copyright 2014 The Kubernetes Authors.
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

# This script finds, caches, and prints a list of all directories that hold
# *.go files.  If any directory is newer than the cache, re-find everything and
# update the cache.  Otherwise use the cached file.

set -o errexit
set -o nounset
set -o pipefail

if [[ -z "${1:-}" ]]; then
    echo "usage: $0 <cache-file>"
    exit 1
fi
CACHE="$1"; shift

trap "rm -f '${CACHE}'" HUP INT TERM ERR

# This is a partial 'find' command.  The caller is expected to pass the
# remaining arguments.
#
# Example:
#   kfind -type f -name foobar.go
function kfind() {
    # We want to include the "special" vendor directories which are actually
    # part of the Kubernetes source tree (./staging/*) but we need them to be
    # named as their ./vendor/* equivalents.  Also, we  do not want all of
    # ./vendor or even all of ./vendor/k8s.io.
    find -H .                      \
        \(                         \
        -not \(                    \
            \(                     \
                -path ./vendor -o  \
                -path ./_\* -o     \
                -path ./.\* -o     \
                -path ./docs       \
            \) -prune              \
        \)                         \
        \)                         \
        "$@"                       \
        | sed 's|^./staging/src|vendor|'
}

NEED_FIND=true
# It's *significantly* faster to check whether any directories are newer than
# the cache than to blindly rebuild it.
if [[ -f "${CACHE}" ]]; then
    N=$(kfind -type d -newer "${CACHE}" -print -quit | wc -l)
    [[ "${N}" == 0 ]] && NEED_FIND=false
fi
mkdir -p $(dirname "${CACHE}")
if $("${NEED_FIND}"); then
    kfind -type f -name \*.go  \
        | sed 's|/[^/]*$||'    \
        | sed 's|^./||'        \
        | LC_ALL=C sort -u     \
        > "${CACHE}"
fi
cat "${CACHE}"
