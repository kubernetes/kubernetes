#!/bin/sh
set -e

docker build -t kubelet-rpm-builder .
echo "Cleaning output directory..."
sudo rm -rf output/*
mkdir -p output
docker run -ti --rm -v $PWD/output/:/root/rpmbuild/RPMS/ kubelet-rpm-builder $1
sudo chown -R $USER $PWD/output

echo
echo "----------------------------------------"
echo
echo "RPMs written to: "
ls $PWD/output/*/
echo
echo "Yum repodata written to: "
ls $PWD/output/*/repodata/
