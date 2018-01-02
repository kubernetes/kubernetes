#!/usr/bin/env bash

set -e

function raise()
{
    kill -$1 0
}

trap "raise SIGINT" SIGINT
make $1
