#!/bin/bash

SWAKS="/root/swaks"

# Bump this whenever a release branch is created from master
MASTER_VERSION=0.10.0

# send_failure_notification sends an e-mail with a build failure notification.
function send_failure_notification {
    smtp=$1
    user=$2
    password=$3
    to=$4
    version=$5
    $SWAKS --auth \
        --server $smtp \
        --au $user \
        --ap $password \
        --to $to \
        --h-Subject: "Nightly build has FAILED" \
        --body "The nightly build has failed, version: $version"
}

if [ $# -lt 4 ]; then
    echo "$0 <smtp server> <user> <password> <to> [RACE_ENABLED]"
    exit 1
fi
SMTP=$1
USER=$2
PASSWORD=$3
TO=$4
RACE_ENABLED=$5

if [ -n "$RACE_ENABLED" ]; then
    race="-x"
    echo "Race-detection build enabled."
fi

REPO_DIR=`mktemp -d`
echo "Using $REPO_DIR for all work..."

cd $REPO_DIR
export GOPATH=`pwd`
mkdir -p $GOPATH/src/github.com/influxdata
cd $GOPATH/src/github.com/influxdata
git clone https://github.com/influxdata/influxdb.git

cd $GOPATH/src/github.com/influxdata/influxdb
VERSION="$MASTER_VERSION-nightly-`git log --pretty=format:'%h' -n 1`"
NIGHTLY_BUILD=true ./package.sh $race $VERSION

if [ $? -ne 0 ]; then
    # Send notification e-mail.
    send_failure_notification $SMTP $USER $PASSWORD $TO $VERSION
fi

rm -rf $REPO_DIR
