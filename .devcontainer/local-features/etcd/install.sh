#!/usr/bin/env bash
# Copyright 2022 The Kubernetes Authors.
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

# Only supports Linux

set -eux

ETCD_VERSION="${VERSION:-"latest"}"

# Figure out correct version of a three part version number is not passed
find_version_from_git_tags() {
    local variable_name=$1
    local requested_version=${!variable_name}
    if [ "${requested_version}" = "none" ]; then return; fi
    local repository=$2
    local prefix=${3:-"tags/v"}
    local separator=${4:-"."}
    local last_part_optional=${5:-"false"}
    if [ "$(echo "${requested_version}" | grep -o "." | wc -l)" != "2" ]; then
        local escaped_separator=${separator//./\\.}
        local last_part
        if [ "${last_part_optional}" = "true" ]; then
            last_part="(${escaped_separator}[0-9]+)?"
        else
            last_part="${escaped_separator}[0-9]+"
        fi
        local regex="${prefix}\\K[0-9]+${escaped_separator}[0-9]+${last_part}$"
        # shellcheck disable=SC2155
        local version_list="$(git ls-remote --tags "${repository}" | grep -oP "${regex}" | tr -d ' ' | tr "${separator}" "." | sort -rV)"
        if [ "${requested_version}" = "latest" ] || [ "${requested_version}" = "current" ] || [ "${requested_version}" = "lts" ]; then
            # shellcheck disable=SC2086
            declare -g ${variable_name}="$(echo "${version_list}" | head -n 1)"
        else
            set +e
                # shellcheck disable=SC2086
                declare -g ${variable_name}="$(echo "${version_list}" | grep -E -m 1 "^${requested_version//./\\.}([\\.\\s]|$)")"
            set -e
        fi
    fi
    if [ -z "${!variable_name}" ] || ! echo "${version_list}" | grep "^${!variable_name//./\\.}$" > /dev/null 2>&1; then
        echo -e "Invalid ${variable_name} value: ${requested_version}\nValid values:\n${version_list}" >&2
        exit 1
    fi
    # shellcheck disable=SC1079
    echo "${variable_name}=${!variable_name}"
}

apt-get update
apt-get -y install --no-install-recommends curl tar

# Get closest match for version number specified
find_version_from_git_tags ETCD_VERSION "https://github.com/etcd-io/etcd"

# Installs etcd in ./third_party/etcd
echo "Installing etcd ${ETCD_VERSION}..."

architecture="$(uname -m)"
case "${architecture}" in
    "x86_64") architecture="amd64"
    ;;
    "aarch64" | "armv8*") architecture="arm64"
    ;;
    *) echo "(!) Architecture ${architecture} unsupported";
    exit 1
    ;;
esac

# shellcheck disable=SC1079
FILE_NAME="etcd-v${ETCD_VERSION}-linux-${architecture}.tar.gz"
curl -sSL -o "${FILE_NAME}" "https://github.com/coreos/etcd/releases/download/v${ETCD_VERSION}/${FILE_NAME}"
tar xzf "${FILE_NAME}"

# shellcheck disable=SC1073
mv "etcd-v${ETCD_VERSION}-linux-amd64" /usr/local/etcd
rm -rf "${FILE_NAME}"

# Installs etcd in /usr/bin so we don't have to futz with the path.
install -m755 /usr/local/etcd/etcd /usr/local/bin/etcd
install -m755 /usr/local/etcd/etcdctl /usr/local/bin/etcdctl
install -m755 /usr/local/etcd/etcdutl /usr/local/bin/etcdutl