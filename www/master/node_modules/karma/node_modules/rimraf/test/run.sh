#!/bin/bash
set -e
for i in test-*.js; do
  echo -n $i ...
  bash setup.sh
  node $i
  ! [ -d target ]
  echo "pass"
done
rm -rf target
