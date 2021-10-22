#!/bin/bash

# Copyright 2019 Google LLC.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

today=$(date +%Y%m%d)

sed -i -r -e 's/const Repo = "([0-9]{8})"/const Repo = "'$today'"/' $GOFILE

