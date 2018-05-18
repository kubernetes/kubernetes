/*
Copyright 2018 The Kubernetes Authors.

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

package version

import (
	"testing"
)

func TestCompareKubeAwareVersionStrings(t *testing.T) {
	tests := []*struct {
		v1, v2          string
		expectedGreater bool
	}{
		{"v1", "v2", false},
		{"v2", "v1", true},
		{"v10", "v2", true},
		{"v1", "v2alpha1", true},
		{"v1", "v2beta1", true},
		{"v1alpha2", "v1alpha1", true},
		{"v1beta1", "v2alpha3", true},
		{"v1alpha10", "v1alpha2", true},
		{"v1beta10", "v1beta2", true},
		{"foo", "v1beta2", false},
		{"bar", "foo", true},
		{"version1", "version2", true},  // Non kube-like versions are sorted alphabetically
		{"version1", "version10", true}, // Non kube-like versions are sorted alphabetically
	}

	for _, tc := range tests {
		if e, a := tc.expectedGreater, CompareKubeAwareVersionStrings(tc.v1, tc.v2) > 0; e != a {
			if e {
				t.Errorf("expected %s to be greater than %s", tc.v1, tc.v2)
			} else {
				t.Errorf("expected %s to be less than than %s", tc.v1, tc.v2)
			}
		}
	}
}
