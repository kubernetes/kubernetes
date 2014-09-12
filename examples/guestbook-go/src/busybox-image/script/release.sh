#!/bin/bash
# Usage: ./release.sh [TAG]

set +e

DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

echo " ---> Building..."
sh $DIR/build.sh

echo " ---> Pushing gurpartap/guestbook-example:${1:-latest}..."
sh $DIR/push.sh $1

echo " ---> Cleaning up..."
sh $DIR/clean.sh

echo " ---> Done."
