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

# A set of Cloud Files defaults for which Kubernetes releases will be uploaded to

# Make sure swiftly is installed and available
if [ "$(which swiftly)" == "" ]; then
  echo "release/rackspace/config.sh: Couldn't find swiftly in PATH. Please install swiftly:"
  echo -e "\tpip install swiftly"
  exit 1
fi

CONTAINER="kubernetes-releases-${OS_USERNAME}"

TAR_FILE=master-release.tgz
