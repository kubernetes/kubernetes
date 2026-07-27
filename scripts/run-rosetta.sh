#!/bin/bash
#
# Run jsii-rosetta on all jsii packages, using the S3 build cache if available.
#
#       Usage: run-rosetta [--infuse] [--no-check-conflicts] [--pkgs-from PKGSFILE]
#
# Performs three steps, in that order:
#
# 1. Run `rosetta extract` to read and translate all examples from all JSII
#    assemblies.
#
# 2. Run `rosetta infuse` to traverse all examples we have, and copy them
#    to classes that don't have an example yet.
#
# 3. Run `cdk-generate-synthetic-examples` to find all types that *still*
#    don't have associated examples, and generate synthetic examples.
#
# By default, loads all JSII packages into a single TypeSystem using jsii-tree
# to detect version conflicts early. This is skipped when --infuse is used,
# since cdk-generate-synthetic-examples already loads all packages and will
# fail on conflicts. Use --no-check-conflicts to disable explicitly.
#
# If you already have a file with a list of all the JSII package directories
# in it, pass it as the first argument. Otherwise, this script will run
# 'list-packages' to determine a list itself.
set -eu
scriptdir=$(cd $(dirname $0) && pwd)

ROSETTA=${ROSETTA:-yarn run rosetta}

infuse=false
check_conflicts=true
jsii_pkgs_file=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --infuse)
            infuse="true"
            check_conflicts="false"
            ;;
        --no-check-conflicts)
            check_conflicts="false"
            ;;
        --pkgs-from)
            jsii_pkgs_file="$2"
            shift
            ;;
        -h|--help)
            echo "Usage: run-rosetta.sh [--infuse] [--no-check-conflicts] [--pkgs-from FILE]" >&2
            exit 1
            ;;
        *)
            echo "Unrecognized argument: $1" >&2
            exit 1
            ;;
    esac
    shift
done


if [[ "${jsii_pkgs_file}" = "" ]]; then
    echo "Collecting package list..." >&2
    TMPDIR=${TMPDIR:-$(dirname $(mktemp -u))}
    node $scriptdir/list-packages $TMPDIR/jsii.txt $TMPDIR/nonjsii.txt
    jsii_pkgs_file=$TMPDIR/jsii.txt
fi

mkdir -p $HOME/.s3buildcache
batch_size=100
rosetta_cache_file=$HOME/.s3buildcache/rosetta-cache.tabl.json
extract_opts=""
if $infuse; then
    extract_opts="--infuse"
fi

#----------------------------------------------------------------------
# Rosetta Debug settings

export TIMING="1"
export DEBUG_TYPE_FINGERPRINTS=$HOME/.s3buildcache/type-fingerprints.txt

#----------------------------------------------------------------------

echo "💎 Extracting code samples" >&2
time $ROSETTA extract \
    --batch $batch_size \
    --compile \
    --verbose \
    --compress-tablet \
    --cache ${rosetta_cache_file} \
    ${extract_opts} \
    $(cat $jsii_pkgs_file)

if $infuse; then
    echo "💎 Generating synthetic examples for the remainder" >&2
    time yarn run synthetic-examples \
        $(cat $jsii_pkgs_file)

    time $ROSETTA extract \
        --batch $batch_size \
        --compile \
        --verbose \
        --compress-tablet \
        --cache ${rosetta_cache_file} \
        $(cat $jsii_pkgs_file)
elif $check_conflicts; then
    echo "💎 Checking for JSII type system conflicts" >&2
    time $scriptdir/../node_modules/.bin/jsii-tree --no-validate $(cat $jsii_pkgs_file) > /dev/null
fi

time $ROSETTA trim-cache \
    --verbose \
    ${rosetta_cache_file} \
    $(cat $jsii_pkgs_file)
