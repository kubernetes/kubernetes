#!/bin/bash

REPO_ROOT="$(realpath "$(dirname $0)/..")"

result=0

dirs=("pkg" "cmd")

for dir in ${dirs}; do
  for file in $(grep -r -l "" "${REPO_ROOT}/${dir}/" | grep "[.]go"); do
    if [[ "$(${REPO_ROOT}/hooks/boilerplate.sh "${file}")" -eq "0" ]]; then
      echo "Boilerplate header is wrong for: ${file}"
      result=1
    fi
  done
done

exit ${result}
