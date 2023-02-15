#!/bin/bash
set -e

mkdir -p bin dist
if [ -e ./scripts/$1 ]; then
    ./scripts/"$@"
else
    exec "$@"
fi

chown -R $DAPPER_UID:$DAPPER_GID .
