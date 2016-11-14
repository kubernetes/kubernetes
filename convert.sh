#!/bin/bash

# Copyright 2016 The Kubernetes Authors.
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

set -euxo pipefail

#echo "+ output to $files_to_convert"
# find . -type f -name "*.go" \( -path './pkg/*' -o -path './federation/*' -o -path './test/*' \) > "${files_to_convert}"

files_to_convert=$( mktemp )
grep -lR --include=*.go -E "/unversioned" pkg test federation plugin cmd > "${files_to_convert}"

cat ${files_to_convert} | xargs -n1 sed -i "s|unversioned\\.GroupVersion|schema\\.GroupVersion|g"
cat ${files_to_convert} | xargs -n1 sed -i "s|unversioned\\.GroupResource|schema\\.GroupResource|g"
cat ${files_to_convert} | xargs -n1 sed -i "s|unversioned\\.GroupKind|schema\\.GroupKind|g"
cat ${files_to_convert} | xargs -n1 sed -i "s|unversioned\\.ObjectKind|schema\\.ObjectKind|g"
cat ${files_to_convert} | xargs -n1 sed -i "s|unversioned\\.FromAPIVersionAndKind|schema\\.FromAPIVersionAndKind|g"
cat ${files_to_convert} | xargs -n1 sed -i "s|unversioned\\.EmptyObjectKind|schema\\.EmptyObjectKind|g"
cat ${files_to_convert} | xargs -n1 sed -i "s|unversioned\\.ParseGroupVersion|schema\\.ParseGroupVersion|g"
cat ${files_to_convert} | xargs -n1 sed -i "s|unversioned\\.ParseResourceArg|schema\\.ParseResourceArg|g"
echo pkg/client/typed/discovery/helper_blackbox_test.go | xargs -n1 sed -i "s|uapi\\.GroupVersion|schema\\.GroupVersion|g"
cat ${files_to_convert} | xargs -n1 sed -i "s|schema\\.GroupVersionForDiscovery|unversioned\\.GroupVersionForDiscovery|g"
goimports -w ./pkg ./federation ./test ./plugin ./cmd
echo test/e2e_node/apparmor_test.go | xargs -n1 sed -i "s|k8s\\.io/kubernetes/pkg/runtime/schema|k8s\\.io/client-go/pkg/runtime/schema|g"
