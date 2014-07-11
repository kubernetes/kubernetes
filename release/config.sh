# Copyright 2014 Google Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# A set of defaults for Kubernetes releases

if [ "$(which gcloud)" == "" ]; then
  echo "Couldn't find gcloud in PATH"
  exit 1
fi

if [ -n "$(gcloud auth list 2>&1 | grep 'No credentialed accounts')" ]; then
    gcloud auth login
fi

PROJECT=$(gcloud config list project | tail -n 1 | cut -f 3 -d ' ')

if [ ! -n "$PROJECT" ]; then
    echo "Default project is not set."
    echo "Please run gcloud config set project <project>"
    exit 1
fi

if which md5 > /dev/null 2>&1; then
  HASH=$(md5 -q -s $PROJECT)
else
  HASH=$(echo -n "$PROJECT" | md5sum)
fi
HASH=${HASH:0:5}
RELEASE_BUCKET=${RELEASE_BUCKET-gs://kubernetes-releases-$HASH/}
RELEASE_PREFIX=${RELEASE_PREFIX-devel/$USER/}
RELEASE_NAME=${RELEASE_NAME-r$(date -u +%Y%m%d-%H%M%S)}

# This is a 'soft link' to the release in question.  It is a single line file to
# the full GS path for a release.
RELEASE_TAG=${RELEASE_TAG-testing}

RELEASE_TAR_FILE=master-release.tgz

RELEASE_FULL_PATH=$RELEASE_BUCKET$RELEASE_PREFIX$RELEASE_NAME
RELEASE_FULL_TAG_PATH=$RELEASE_BUCKET$RELEASE_PREFIX$RELEASE_TAG

# Takes a release path ($1 if passed, otherwise $RELEASE_FULL_TAG_PATH) and
# computes the normalized release path. Results are stored in
# $RELEASE_NORMALIZED.  Returns 0 if a valid release can be found.
function normalize_release() {
  RELEASE_NORMALIZED=${1-$RELEASE_FULL_TAG_PATH}

  # First test to see if there is a valid release at this path.
  if gsutil -q stat $RELEASE_NORMALIZED/$RELEASE_TAR_FILE; then
    return 0
  fi

  # Check if this is a simple file.  If so, read it and use the result as the
  # new RELEASE_NORMALIZED.
  if gsutil -q stat $RELEASE_NORMALIZED; then
    RELEASE_NORMALIZED=$(gsutil -q cat $RELEASE_NORMALIZED)
    normalize_release $RELEASE_NORMALIZED
    return
  fi
  return 1
}

# Sets a tag ($1) to a release ($2)
function set_tag() {
  echo $2 | gsutil -q cp - $1

  gsutil -q setmeta -h "Cache-Control:private, max-age=0, no-transform" $1
  make_public_readable $1
}

# Makes a GCS object ($1) publicly readable
function make_public_readable() {
  # Ideally we'd run the command below.  But this is currently broken in the
  # newest version of gsutil.  Instead, download the ACL and edit the json
  # quickly.

  # gsutil -q acl ch -g AllUsers:R $1

  TMPFILE=$(mktemp -t release 2>/dev/null || mktemp -t release.XXXX)

  gsutil -q acl get $1 \
    | python $(dirname $0)/make-public-gcs-acl.py \
    > $TMPFILE
    gsutil -q acl set $TMPFILE $RELEASE_FULL_PATH/$x

  rm $TMPFILE
}
