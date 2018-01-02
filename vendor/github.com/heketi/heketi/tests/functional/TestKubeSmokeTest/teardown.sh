#!/bin/sh

TOP=../../..
CURRENT_DIR=`pwd`
RESOURCES_DIR=$CURRENT_DIR/resources
FUNCTIONAL_DIR=${CURRENT_DIR}/..

source ${FUNCTIONAL_DIR}/lib.sh


if [ -x /usr/local/bin/minikube ] ; then
    minikube stop
    minikube delete
fi
rm -rf $RESOURCES_DIR > /dev/null
