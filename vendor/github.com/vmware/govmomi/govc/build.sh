#!/bin/bash -e

git_version=$(git describe)
if git_status=$(git status --porcelain 2>/dev/null) && [ -n "${git_status}" ]; then
  git_version="${git_version}-dirty"
fi

ldflags="-X github.com/vmware/govmomi/govc/version.gitVersion=${git_version}"

BUILD_OS=${BUILD_OS:-darwin linux windows freebsd}
BUILD_ARCH=${BUILD_ARCH:-386 amd64}

for os in ${BUILD_OS}; do
  export GOOS="${os}"
  for arch in ${BUILD_ARCH}; do
    export GOARCH="${arch}"

    out="govc_${os}_${arch}"
    if [ "${os}" == "windows" ]; then
      out="${out}.exe"
    fi

    set -x
    go build \
      -o="${out}" \
      -pkgdir="./_pkg" \
      -compiler='gc' \
      -ldflags="${ldflags}" \
      github.com/vmware/govmomi/govc &
    set +x
  done
done

wait
