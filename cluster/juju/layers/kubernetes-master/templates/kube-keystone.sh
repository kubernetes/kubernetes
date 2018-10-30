# Copyright 2018 The Kubernetes Authors.
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

# Replace with your public address and port for keystone
export OS_AUTH_URL="{{ protocol }}://{{ address }}:{{ port }}/v{{ version }}"
#export OS_PROJECT_NAME=k8s
#export OS_DOMAIN_NAME=k8s
#export OS_USERNAME=myuser
#export OS_PASSWORD=secure_pw
get_keystone_token() {
  data='{ "auth": {
    "identity": {
      "methods": ["password"],
      "password": {
        "user": {
          "name": "'"${OS_USERNAME}"'",
          "domain": { "name": "'"${OS_DOMAIN_NAME}"'" },
          "password": "'"${OS_PASSWORD}"'"
        }
      }
    }
  }
}'
  token=$(curl -s -i -H "Content-Type: application/json" -d "${data}" "${OS_AUTH_URL}/auth/tokens" |grep 'X-Subject-Token')
  if [ -z "$token" ]; then
    echo "Invalid authentication information"
  else
    echo $(echo ${token} | awk -F ': ' '{print $2}')
  fi
}
echo "Function get_keystone_token created. Type get_keystone_token in order to generate a login token for the Kubernetes dashboard."

