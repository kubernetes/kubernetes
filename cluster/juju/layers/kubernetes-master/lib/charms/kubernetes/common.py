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

from time import sleep


def get_version(bin_name):
    """Get the version of an installed Kubernetes binary.

    :param str bin_name: Name of binary
    :return: 3-tuple version (maj, min, patch)

    Example::

        >>> `get_version('kubelet')
        (1, 6, 0)

    """
    cmd = '{} --version'.format(bin_name).split()
    version_string = subprocess.check_output(cmd).decode('utf-8')
    return tuple(int(q) for q in re.findall("[0-9]+", version_string)[:3])


def retry(times, delay_secs):
    """ Decorator for retrying a method call.

    Args:
        times: How many times should we retry before giving up
        delay_secs: Delay in secs

    Returns: A callable that would return the last call outcome
    """

    def retry_decorator(func):
        """ Decorator to wrap the function provided.

        Args:
            func: Provided function should return either True od False

        Returns: A callable that would return the last call outcome

        """
        def _wrapped(*args, **kwargs):
            res = func(*args, **kwargs)
            attempt = 0
            while not res and attempt < times:
                sleep(delay_secs)
                res = func(*args, **kwargs)
                if res:
                    break
                attempt += 1
            return res
        return _wrapped

    return retry_decorator
