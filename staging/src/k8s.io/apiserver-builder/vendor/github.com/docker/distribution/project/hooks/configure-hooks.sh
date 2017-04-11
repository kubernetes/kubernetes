#!/bin/sh

cd $(dirname $0)

REPO_ROOT=$(git rev-parse --show-toplevel)
RESOLVE_REPO_ROOT_STATUS=$?
if [ "$RESOLVE_REPO_ROOT_STATUS" -ne "0" ]; then
	echo -e "Unable to resolve repository root. Error:\n$REPO_ROOT" > /dev/stderr
	exit $RESOLVE_REPO_ROOT_STATUS
fi

set -e
set -x

# Just in case the directory doesn't exist
mkdir -p $REPO_ROOT/.git/hooks

ln -f -s $(pwd)/pre-commit $REPO_ROOT/.git/hooks/pre-commit