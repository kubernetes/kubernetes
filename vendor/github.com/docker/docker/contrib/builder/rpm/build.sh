#!/bin/bash
set -e

cd "$(dirname "$(readlink -f "$BASH_SOURCE")")"

set -x
./generate.sh
for d in */; do
	docker build -t "dockercore/builder-rpm:$(basename "$d")" "$d"
done
