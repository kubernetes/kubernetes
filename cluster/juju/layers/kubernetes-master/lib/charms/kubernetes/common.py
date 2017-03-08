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
