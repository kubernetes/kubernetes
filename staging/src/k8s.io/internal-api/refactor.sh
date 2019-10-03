#!/bin/bash

git mv pkg/api staging/src/k8s.io/internal-api/
git mv pkg/apis staging/src/k8s.io/internal-api/
git add .
git commit -m "refactor: Move internal APIs to new location"

find staging/src/k8s.io/internal-api -name *.go | xargs -L1 sed -i "s|k8s.io/kubernetes/pkg/apis|k8s.io/internal-api/apis|g"
find staging/src/k8s.io/internal-api -name *.go | xargs -L1 sed -i "s|k8s.io/kubernetes/pkg/api|k8s.io/internal-api/api|g"
git add .
git commit -m "refactor: Update package references inside staging API"

for i in pkg cmd test plugin; do
  find $i -name *.go | xargs -L1 sed -i "s|k8s.io/kubernetes/pkg/apis|k8s.io/internal-api/apis|g"
  find $i -name *.go | xargs -L1 sed -i "s|k8s.io/kubernetes/pkg/api|k8s.io/internal-api/api|g"
done
git add .
git commit -m "refactor: Update package references outside staging API"