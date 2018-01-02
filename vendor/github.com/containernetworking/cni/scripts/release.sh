#!/usr/bin/env bash
set -xe

SRC_DIR="${SRC_DIR:-$PWD}"
BUILDFLAGS="-a --ldflags '-extldflags \"-static\"'"

TAG=$(git describe --tags --dirty)
RELEASE_DIR=release-${TAG}

OUTPUT_DIR=bin

# Always clean first
rm -Rf ${SRC_DIR}/${RELEASE_DIR}
mkdir -p ${SRC_DIR}/${RELEASE_DIR}

docker run -i -v ${SRC_DIR}:/opt/src --rm golang:1.8-alpine \
/bin/sh -xe -c "\
    apk --no-cache add bash tar;
    cd /opt/src; umask 0022;
    for arch in amd64 arm arm64 ppc64le s390x; do \
        CGO_ENABLED=0 GOARCH=\$arch ./build.sh ${BUILDFLAGS}; \
        for format in tgz; do \
            FILENAME=cni-\$arch-${TAG}.\$format; \
            FILEPATH=${RELEASE_DIR}/\$FILENAME; \
            tar -C ${OUTPUT_DIR} --owner=0 --group=0 -caf \$FILEPATH .; \
            if [ \"\$arch\" == \"amd64\" ]; then \
                cp \$FILEPATH ${RELEASE_DIR}/cni-${TAG}.\$format; \
            fi; \
        done; \
    done;
    cd ${RELEASE_DIR};
      for f in *.tgz; do sha1sum \$f > \$f.sha1; done;
      for f in *.tgz; do sha256sum \$f > \$f.sha256; done;
      for f in *.tgz; do sha512sum \$f > \$f.sha512; done;
    cd ..
    chown -R ${UID} ${OUTPUT_DIR} ${RELEASE_DIR}"
