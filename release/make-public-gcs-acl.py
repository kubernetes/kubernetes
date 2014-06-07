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

# This is a quick script that adds AllUsers as READER to a JSON file
# representing an ACL on a GCS object.  This is a quick workaround for a bug in
# gsutil.
import json
import sys

acl = json.load(sys.stdin)
acl.append({
    "entity": "allUsers",
    "role": "READER"
  })
json.dump(acl, sys.stdout)
