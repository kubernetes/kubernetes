#!/bin/sh
# Entrypoint for the build container to create the rpms and yum repodata:
# Usage: ./entry.sh GOARCH/RPMARCH,GOARCH/RPMARCH,....

set -e

declare -a ARCHS

if [ $# -gt 0 ]; then
  IFS=','; ARCHS=($1); unset IFS;
else
  #GOARCH/RPMARCH
  ARCHS=(
    amd64/x86_64
    arm/armhfp
    arm64/aarch64
    ppc64le/ppc64le
    s390x/s390x
  )
fi

for ARCH in ${ARCHS[@]}; do
  IFS=/ read GOARCH RPMARCH<<< ${ARCH}; unset IFS;
  SRC_PATH="/root/rpmbuild/SOURCES/${RPMARCH}"
  mkdir -p ${SRC_PATH}
  cp -r /root/rpmbuild/SPECS/* ${SRC_PATH}
  echo "Building RPM's for ${GOARCH}....."
  sed -i "s/\%global ARCH.*/\%global ARCH ${GOARCH}/" ${SRC_PATH}/kubelet.spec
  # Download sources if not already available
  cd ${SRC_PATH} && spectool -gf kubelet.spec
  /usr/bin/rpmbuild --target ${RPMARCH} --define "_sourcedir ${SRC_PATH}" -bb ${SRC_PATH}/kubelet.spec
  mkdir -p /root/rpmbuild/RPMS/${RPMARCH}
  createrepo -o /root/rpmbuild/RPMS/${RPMARCH}/ /root/rpmbuild/RPMS/${RPMARCH}
done
