/*
Copyright 2022 The Kubernetes Authors.

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

package helper

import (
	"testing"

	v1 "k8s.io/api/core/v1"
)

func TestDoNotScheduleTaintsFilterFunc(t *testing.T) {
	tests := []struct {
		name     string
		taint    *v1.Taint
		expected bool
	}{
		{
			name: "should include the taints with NoSchedule effect",
			taint: &v1.Taint{
				Effect: v1.TaintEffectNoSchedule,
			},
			expected: true,
		},
		{
			name: "should include the taints with NoExecute effect",
			taint: &v1.Taint{
				Effect: v1.TaintEffectNoExecute,
			},
			expected: true,
		},
		{
			name: "should not include the taints with PreferNoSchedule effect",
			taint: &v1.Taint{
				Effect: v1.TaintEffectPreferNoSchedule,
			},
			expected: false,
		},
	}

	filterPredicate := DoNotScheduleTaintsFilterFunc()

	for i := range tests {
		test := tests[i]
		t.Run(test.name, func(t *testing.T) {
			if got := filterPredicate(test.taint); got != test.expected {
				t.Errorf("unexpected result, expected %v but got %v", test.expected, got)
			}
		})
	}
}
