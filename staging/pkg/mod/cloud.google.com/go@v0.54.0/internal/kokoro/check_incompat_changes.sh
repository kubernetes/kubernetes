#!/usr/bin/env bash
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Skipped until https://github.com/golang/go/issues/34849 is resolved.
exit 0

# Display commands being run
set -x

# Only run apidiff checks on go1.12 (we only need it once).
if [[ `go version` != *"go1.12"* ]]; then
    exit 0
fi

if git log -1 | grep BREAKING_CHANGE_ACCEPTABLE; then
  exit 0
fi

go install golang.org/x/exp/cmd/apidiff

# We compare against master@HEAD. This is unfortunate in some cases: if you're
# working on an out-of-date branch, and master gets some new feature (that has
# nothing to do with your work on your branch), you'll get an error message.
# Thankfully the fix is quite simple: rebase your branch.
git clone https://code.googlesource.com/gocloud /tmp/gocloud

MANUALS="bigquery bigtable datastore firestore pubsub spanner storage logging"
STABLE_GAPICS="container/apiv1 dataproc/apiv1 iam iam/admin/apiv1 iam/credentials/apiv1 kms/apiv1 language/apiv1 logging/apiv2 logging/logadmin pubsub/apiv1 spanner/apiv1 translate/apiv1 vision/apiv1"
for dir in $MANUALS $STABLE_GAPICS; do
  pkg="cloud.google.com/go/$dir"
  echo "Testing $pkg"

  cd /tmp/gocloud
  apidiff -w /tmp/pkg.master $pkg
  cd - > /dev/null

  # TODO(deklerk): there's probably a nicer way to do this that doesn't require
  # two invocations
  if ! apidiff -incompatible /tmp/pkg.master $pkg | (! read); then
    echo "Detected incompatible API changes between master@HEAD and current state:"
    apidiff -incompatible /tmp/pkg.master $pkg
    exit 1
  fi
done
