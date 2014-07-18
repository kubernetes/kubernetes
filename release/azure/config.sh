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

AZ_LOCATION='West US'
AZ_SSH_KEY=$HOME/.ssh/azure
AZ_SSH_CERT=$HOME/.ssh/azure.pem
AZ_IMAGE=b39f27a8b8c64d52b05eac6a62ebad85__Ubuntu-14_04-LTS-amd64-server-20140618.1-en-us-30GB
AZ_SUBNET=Subnet-1
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
  hsh=$(md5 -q -s $account)
else
  hsh=$(echo -n "$account" | md5sum)
fi
hsh=${hsh:0:7}

STG_ACCOUNT=kube$hsh

AZ_VNET=kube-$hsh
AZ_CS=kube-$hsh

CONTAINER=kube-$TAG

FULL_URL="https://${STG_ACCOUNT}.blob.core.windows.net/$CONTAINER/master-release.tgz"
