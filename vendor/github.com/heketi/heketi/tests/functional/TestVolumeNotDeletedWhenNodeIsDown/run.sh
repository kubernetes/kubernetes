#!/bin/sh

CURRENT_DIR=`pwd`
HEKETI_SERVER_BUILD_DIR=../../..
FUNCTIONAL_DIR=${CURRENT_DIR}/..
HEKETI_SERVER=${FUNCTIONAL_DIR}/heketi-server

source ${FUNCTIONAL_DIR}/lib.sh

functional_tests

