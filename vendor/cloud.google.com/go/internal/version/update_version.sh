#!/bin/bash

today=$(date +%Y%m%d)

sed -i -r -e 's/const Repo = "([0-9]{8})"/const Repo = "'$today'"/' $GOFILE

