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

if which md5 > /dev/null 2>&1; then
    MD5_FUNC=md5
else
    MD5_FUNC=md5sum
fi

function json_val () {
    python -c 'import json,sys;obj=json.load(sys.stdin);print obj'$1''
}

INSTANCE_PREFIX=kubernetes

AWS_HASH=$(aws --output json iam list-access-keys | json_val '["AccessKeyMetadata"][0]["AccessKeyId"]' | $MD5_FUNC)
AWS_HASH=${AWS_HASH:0:5}
RELEASE_BUCKET=${RELEASE_BUCKET-s3://kubernetes-releases-$AWS_HASH/}
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
  if aws s3 ls $RELEASE_NORMALIZED/$RELEASE_TAR_FILE | grep $RELEASE_TAR_FILE > /dev/null; then
    return 0
  fi

  # Check if this is a simple file.  If so, read it and use the result as the
  # new RELEASE_NORMALIZED.
  if aws s3 ls $RELEASE_NORMALIZED | grep $RELEASE_TAG > /dev/null; then
    RELEASE_NORMALIZED=$(aws s3 cp $RELEASE_NORMALIZED >(cat) > /dev/null)
    normalize_release $RELEASE_NORMALIZED
    RELEASE_FULL_HTTP_PATH=${RELEASE_NORMALIZED/s3:\/\//https:\/\/s3-$ZONE.amazonaws.com/}
    return
  fi
  return 1
}

# Sets a tag ($1) to a release ($2)
function set_tag() {
  TMPFILE=$(mktemp -t release_tag 2>/dev/null || mktemp -t release_tag.XXXX)
  echo $2 > $TMPFILE
  aws s3 cp $TMPFILE $1 > /dev/null
}