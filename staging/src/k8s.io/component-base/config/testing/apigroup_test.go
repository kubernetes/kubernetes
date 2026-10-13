/*
Copyright 2019 The Kubernetes Authors.

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

package testing

import (
	"testing"
)

func TestAPIVersionRegexp(t *testing.T) {
	testCases := []struct {
		name       string
		apiversion string
		expected   bool
	}{
		{
			name:       "v1",
			apiversion: "v1",
			expected:   true,
		},
		{
			name:       "v1alpha1",
			apiversion: "v1alpha1",
			expected:   true,
		},
		{
			name:       "v1beta1",
			apiversion: "v1beta1",
			expected:   true,
		},
		{
			name:       "doesn't start with v",
			apiversion: "beta1",
			expected:   false,
		},
		{
			name:       "doesn't end with digit",
			apiversion: "v1alpha",
			expected:   false,
		},
		{
			name:       "doesn't have digit after v",
			apiversion: "valpha1",
			expected:   false,
		},
		{
			name:       "both alpha beta",
			apiversion: "v1alpha1beta1",
			expected:   false,
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			actual := APIVersionRegexp.MatchString(tc.apiversion)
			if actual != tc.expected {
				t.Errorf("APIVersionRegexp expected %v, got %v", tc.expected, actual)
			}
		})
	}
}
