#!/usr/bin/env bash

## Run this script from the root of the docker repository
## to query project stats useful to the maintainers.
## You will need to install `pulls` and `issues` from
## https://github.com/crosbymichael/pulls

set -e

echo -n "Open pulls: "
PULLS=$(pulls | wc -l); let PULLS=$PULLS-1
echo $PULLS

echo -n "Pulls alru: "
pulls alru

echo -n "Open issues: "
ISSUES=$(issues list | wc -l); let ISSUES=$ISSUES-1
echo $ISSUES

echo -n "Issues alru: "
issues alru
