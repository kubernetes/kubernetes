#!/bin/bash

set -o errexit
set -o nounset

# DEPRECATED:
# The functionality has been moved to migrate binary and this script
# if left for backward compatibility with previous manifests. It will be
# removed in the future.

DATA_DIRECTORY=${DATA_DIRECTORY:-/var/etcd/data/}
/usr/local/bin/backup
