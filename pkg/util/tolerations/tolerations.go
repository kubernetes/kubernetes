/*
Copyright 2017 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package tolerations

import (
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/util/tolerations/merge"
)

// VerifyAgainstWhitelist checks if the provided tolerations
// satisfy the provided whitelist and returns true, otherwise returns false
func VerifyAgainstWhitelist(tolerations, whitelist []api.Toleration) bool {
	if len(whitelist) == 0 || len(tolerations) == 0 {
		return true
	}

next:
	for _, t := range tolerations {
		for _, w := range whitelist {
			if merge.IsSuperset(w, t) {
				continue next
			}
		}
		return false
	}

	return true
}
