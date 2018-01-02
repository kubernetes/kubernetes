#!/usr/bin/env bash

set -ex

# Clean up environment after build. It is flushing every assigned IP address via IPAM, umounting every
# mountpoint and removing unused links
function cleanup {
    if [[ "${POSTCLEANUP}" == true ]]; then
        if [[ "${CI}" == true || "${FORCE}" == true ]]; then
            for mp in $(mount | grep rkt | awk '{print $3}' | tac); do
                sudo umount "${mp}"
            done

            for link in $(ip link | grep rkt | cut -d':' -f2 | cut -d'@' -f1); do
                sudo ip link del "${link}"
            done
            sudo rm -rf /var/lib/cni/networks/*
        fi
        sudo rm -rf "${BUILD_DIR}"
    fi
}

# Skip build on demand. It requires the `last-commit` file inside the last commit.
function ciSkip {
    cat last-commit
    echo
    echo "Build skipped as requested in the last commit."
    exit 0
}

# Finds the branching point of two commits. 
# For example, let B and D be two commits, and their ancestry graph as A -> B, A -> C -> D. 
# Given commits B and D, it returns A. 
function getBranchingPoint {
    diff --old-line-format='' --new-line-format='' \
        <(git rev-list --first-parent "${1:-$1}") \
            <(git rev-list --first-parent "${2:-$2}") | head -1
}

# Configure Semaphore CI environment.
function semaphoreCIConfiguration {
    # We might not need to run functional tests or process docs.
    # This is best-effort; || true ensures this does not affect test outcome
    # First, ensure origin is updated - Semaphore can do some weird caching
    git fetch || true
    BRANCHING_POINT=$(getBranchingPoint HEAD origin/master)
    SRC_CHANGES=$(git diff-tree --no-commit-id --name-only -r HEAD..${BRANCHING_POINT} | grep -cEv ${DOC_CHANGE_PATTERN}) || true
    DOC_CHANGES=$(git diff-tree --no-commit-id --name-only -r HEAD..${BRANCHING_POINT} | grep -cE ${DOC_CHANGE_PATTERN}) || true

    # Set up go environment on Semaphore
    if [ -f /opt/change-go-version.sh ]; then
        . /opt/change-go-version.sh
        change-go-version 1.5
        
	# systemd v229 doesn't build on gcc-4.8, set the compiler to gcc-5
        export CC=gcc-5
    fi
}

function checkFlavorValue {
    FLAVORS="coreos host kvm none src fly"
    if [ -z "${RKT_STAGE1_USR_FROM}" ]; then
        set -
        echo "Flavor is not set"
        exit 1
    fi
    if ! [[ "${FLAVORS}" =~ "${RKT_STAGE1_USR_FROM}" ]]; then
        set -
        echo "Unknown flavor: ${RKT_STAGE1_USR_FROM}"
        echo "Available flavors: ${FLAVORS}"
        exit 1
    fi
}

# Parse user provided parameters
function parseParameters {
    while getopts "f:s:r:cxujd" option; do
        case ${option} in
        f)
            RKT_STAGE1_USR_FROM="${OPTARG}"
            ;;
        s)
            if [[ $RKT_STAGE1_USR_FROM == "src" ]]; then
                RKT_STAGE1_SYSTEMD_VER="${OPTARG}"
            else
                echo "Only \`src\` flavor requires systemd version"
            fi
            ;;
        r)
            if [[ $RKT_STAGE1_USR_FROM == "src" ]]; then
                RKT_STAGE1_SYSTEMD_REV="${OPTARG}"
            else
                echo "Only \`src\` flavor requires systemd revision"
            fi
            ;;
        x)
            FORCE=true
            ;;
        u)
            set -
            usage
            exit 0
            ;;
        c)
            PRECLEANUP=true
            POSTCLEANUP=true
            ;;
        j)
            JUSTBUILD=true
            ;;
        d)
            DIRTYBUILD=true
            ;;
        \?)
            set -
            echo "Invalid parameter -${OPTARG}"
            usage
            exit 1
            ;;
        esac
    done
    checkFlavorValue
}

# Configure build
function configure {
    case "${RKT_STAGE1_USR_FROM}" in
        coreos|kvm|fly)
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
                --with-stage1-systemd-version="${RKT_STAGE1_SYSTEMD_VER:-v999}" \
                --with-stage1-systemd-revision="${RKT_STAGE1_SYSTEMD_REV:-master}" \
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
}

# Build rkt and run unit & functional tests
function build {
    ./autogen.sh

    configure

    CORES=$(grep -c ^processor /proc/cpuinfo)
    echo "Running make with ${CORES} threads"
    make "-j${CORES}"
    make manpages bash-completion

    if [[ ${PRECLEANUP} == true ]]; then
        rm -rf "${BUILD_DIR}/tmp/usr_from_${RKT_STAGE1_USR_FROM}"
    fi
    if [[ ${JUSTBUILD} != true ]]; then
        make check
        make "-j${CORES}" clean
    fi
}

# Prepare build directory name
function buildFolder {
    if [[ "${RKT_STAGE1_USR_FROM}" == "src" ]]; then
        POSTFIX="-${RKT_STAGE1_SYSTEMD_VER}"
    fi
    BUILD_DIR="build-rkt-${RKT_STAGE1_USR_FROM}${POSTFIX}"
}

# Detect changes from last commit. If there is no changes, there is no
# need to run build
function detectChanges {
    HEAD=`git rev-parse HEAD`
    MASTER=`git rev-parse origin/master`
    if [[ ${HEAD} == ${MASTER} ]]; then
        SRC_CHANGES=1
        DOC_CHANGES=1
    elif [[ ${SRC_CHANGES} -eq 0 && ${DOC_CHANGES} -eq 0 ]]; then
        echo "No changes detected and HEAD is not origin/master"
        exit 0
    fi
}

# Copy source code into build directory
function copyCode {
    if [[ $(whereis -b rsync | awk '{print $2}') != "" ]]; then
        rsync -aq ../ ${BUILD_DIR} --exclude=".git*" --exclude=builds --exclude-from=../.gitignore
    else
        echo "Cannot find `rsync`, which is required by this shell script"
        exit 1
    fi
}

# Set source code into build directory and enter into it
function setCodeInBuildEnv {
    if [[ ${DIRTYBUILD} == '' ]]; then
        detectChanges
    fi
    copyCode
    pushd "${BUILD_DIR}"
}

# Show usage
function usage {
    echo "build-and-run-tests.sh usage:"
    echo -e "-c\tCleanup"
    echo -e "-d\tUse unsaved changes for build"
    echo -e "-f\tSelect flavor"
    echo -e "-j\tDon't run tests after build"
    echo -e "-s\tSystemd version"
    echo -e "-r\tSystemd revision"
    echo -e "-u\tShow this message"
    echo -e "-x\tUse with '-c' to force cleanup on non-CI systems"
}

# Prepare build environment
function prepareBuildEnv {
    # In case it wasn't cleaned up
    if [ -e "builds/${BUILD_DIR}" ]; then
        sudo rm -rf "builds/${BUILD_DIR}"
    fi
    mkdir -p builds
}

# Run docs scan
function docsScan {
    :
    # echo Changes in docs detected, checking docs.
    # TODO check for broken links
    # TODO check for obvious spelling mistakes:
        # coreos -> CoreOS
        # More?!
}

function main {
    # Skip build if requested
    if test -e ci-skip ; then
        ciSkip
    fi

    SRC_CHANGES=1 # run functional tests by default
    DOC_CHANGES=1 # process docs by default

    parseParameters "${@}"

    DOC_CHANGE_PATTERN="\
            -e ^Documentation/ \
            -e ^dist/ \
            -e ^logos/ \
            -e ^(MAINTAINERS|LICENSE|DCO)$ \
            -e \.md$\
            -e \.(jpeg|jpg|png|svg)$\
    "

    buildFolder

    # https://semaphoreci.com/docs/available-environment-variables.html
    if [ "${SEMAPHORE-}" == true ] ; then
        semaphoreCIConfiguration
    fi

    prepareBuildEnv
    cd builds
    setCodeInBuildEnv

    if [ ${SRC_CHANGES} -gt 0 ]; then
        build
    fi
    if [ ${DOC_CHANGES} -gt 0 ]; then
        docsScan
    fi
    cleanup
}

main "${@}"
