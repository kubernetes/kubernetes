#!/usr/bin/env python

# Copyright 2015 The Kubernetes Authors.
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

import re
import subprocess

from charmhelpers.core import unitdata

BIN_VERSIONS = 'bin_versions'


def get_version(bin_name):
    """Get the version of an installed Kubernetes binary.

    :param str bin_name: Name of binary
    :return: 3-tuple version (maj, min, patch)

    Example::

        >>> `get_version('kubelet')
        (1, 6, 0)

    """
    db = unitdata.kv()
    bin_versions = db.get(BIN_VERSIONS, {})

    cached_version = bin_versions.get(bin_name)
    if cached_version:
        return tuple(cached_version)

    version = _get_bin_version(bin_name)
    bin_versions[bin_name] = list(version)
    db.set(BIN_VERSIONS, bin_versions)
    return version


def reset_versions():
    """Reset the cache of bin versions.

    """
    db = unitdata.kv()
    db.unset(BIN_VERSIONS)


def _get_bin_version(bin_name):
    """Get a binary version by calling it with --version and parsing output.

    """
    cmd = '{} --version'.format(bin_name).split()
    version_string = subprocess.check_output(cmd).decode('utf-8')
    return tuple(int(q) for q in re.findall("[0-9]+", version_string)[:3])
