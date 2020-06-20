#!/bin/bash

set -o errexit
set -o nounset

DATA_DIRECTORY=${DATA_DIRECTORY:-/var/etcd/data/}
/usr/local/bin/backup
