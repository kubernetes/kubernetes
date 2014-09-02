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

INSTANCE_PREFIX=kubenertes
AZ_LOCATION='West US'
TAG=testing

if [ -z "$(which azure)" ]; then
    echo "Couldn't find azure in PATH"
    echo "  please install with 'npm install azure-cli'"
    exit 1
fi

if [ -z "$(azure account list | grep true)" ]; then
    echo "Default azure account not set"
    echo "  please set with 'azure account set'"
    exit 1
fi

account=$(azure account list | grep true | awk '{ print $2 }')
if which md5 > /dev/null 2>&1; then
  AZ_HSH=$(md5 -q -s $account)
else
  AZ_HSH=$(echo -n "$account" | md5sum)
fi
AZ_HSH=${AZ_HSH:0:7}
AZ_STG=kube$AZ_HSH
CONTAINER=kube-$TAG
FULL_URL="https://${AZ_STG}.blob.core.windows.net/$CONTAINER/master-release.tgz"
