#!/usr/bin/env bash

set -ex

# Skip build if requested
if test -e ci-skip ; then
    cat last-commit
    echo
    echo "Build skipped as requested in the last commit."
    exit 0
fi

SRC_CHANGES=1 # run functional tests by default
DOC_CHANGES=1 # process docs by default
DOC_CHANGE_PATTERN="\
        -e ^Documentation/ \
        -e ^(README|ROADMAP|CONTRIBUTING|CHANGELOG)$ \
        -e \.md$\
"

RKT_STAGE1_USR_FROM="${1}"
RKT_STAGE1_SYSTEMD_VER="${2}"
BUILD_DIR="build-rkt-${RKT_STAGE1_USR_FROM}-${RKT_STAGE1_SYSTEMD_VER}"

# https://semaphoreci.com/docs/available-environment-variables.html
if [ "${SEMAPHORE-}" == true ] ; then
        # We might not need to run functional tests or process docs.
        # This is best-effort; || true ensures this does not affect test outcome
        # First, ensure origin is updated - semaphore can do some weird caching
        git fetch || true
        SRC_CHANGES=$(git diff-tree --no-commit-id --name-only -r HEAD..origin/master | grep -cEv ${DOC_CHANGE_PATTERN}) || true
        DOC_CHANGES=$(git diff-tree --no-commit-id --name-only -r HEAD..origin/master | grep -cE ${DOC_CHANGE_PATTERN}) || true

        # Set up go environment on semaphore
        if [ -f /opt/change-go-version.sh ]; then
            . /opt/change-go-version.sh
            change-go-version 1.5
        fi

        # systemd v229 doesn't build on gcc-4.8, set the compiler to gcc-5
        export CC=gcc-5
fi

HEAD=`git rev-parse HEAD`
MASTER=`git rev-parse origin/master`
if [[ ${HEAD} == ${MASTER} ]]; then
    SRC_CHANGES=1
    DOC_CHANGES=1
elif [[ ${SRC_CHANGES} -eq 0 && ${DOC_CHANGES} -eq 0 ]]; then
    echo "No changes detected and HEAD is not origin/master"
    exit 0
fi

# In case it wasn't cleaned up
if [ -e "${BUILD_DIR}" ]; then
    sudo rm -rf "${BUILD_DIR}"
fi

mkdir -p builds
cd builds

git clone ../ "${BUILD_DIR}"
pushd "${BUILD_DIR}"

if [ ${SRC_CHANGES} -gt 0 ]; then
    echo "Changes in sources detected. Running functional tests."
    ./autogen.sh
    case "${RKT_STAGE1_USR_FROM}" in
        coreos|kvm)
        ./configure --with-stage1-flavors="${RKT_STAGE1_USR_FROM}" \
                --with-stage1-default-flavor="${RKT_STAGE1_USR_FROM}" \
                --enable-functional-tests --enable-tpm=auto \
                --enable-insecure-go
        ;;
        host)
        ./configure --with-stage1-flavors=host \
                --with-default-stage1-flavor=host \
                --enable-functional-tests=auto --enable-tpm=auto \
                --enable-insecure-go
        ;;
        src)
        ./configure --with-stage1-flavors="${RKT_STAGE1_USR_FROM}" \
                --with-stage1-default-flavor="${RKT_STAGE1_USR_FROM}" \
                --with-stage1-systemd-version="${RKT_STAGE1_SYSTEMD_VER}" \
                --enable-functional-tests --enable-tpm=auto \
                --enable-insecure-go
        ;;
        none)
        # Not a flavor per se, so perform a detailed setup for some
        # hypothetical 3rd party stage1 image
        ./configure --with-stage1-default-name="example.com/some-stage1-for-rkt" \
                --with-stage1-default-version="0.0.1" --enable-tpm=auto \
                --enable-insecure-go
        ;;
        *)
        echo "Unknown flavor: ${RKT_STAGE1_USR_FROM}"
        exit 1
        ;;
    esac

    CORES=$(grep -c ^processor /proc/cpuinfo)
    echo "Running make with ${CORES} threads"
    make "-j${CORES}"
    make check
    make "-j${CORES}" clean
fi
if [ ${DOC_CHANGES} -gt 0 ]; then
    :
    # echo Changes in docs detected, checking docs.
    # TODO check for broken links
    # TODO check for obvious spelling mistakes:
        # coreos -> CoreOS
        # More?!
fi

popd

# Make sure there is enough disk space for the next build
sudo rm -rf "${BUILD_DIR}"
