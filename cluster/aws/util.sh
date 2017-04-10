#!/bin/bash

# Copyright 2014 The Kubernetes Authors.
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

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/../..
source "${KUBE_ROOT}/hack/lib/util.sh"

echo -e "${color_red}WARNING${color_norm}: The bash deployment for AWS is obsolete. The" >&2
echo -e "v1.5.x releases are the last to support cluster/kube-up.sh with AWS." >&2
echo "For a list of viable alternatives, see:" >&2
echo >&2
echo "  http://kubernetes.io/docs/getting-started-guides/aws/" >&2
echo >&2
exit 1
