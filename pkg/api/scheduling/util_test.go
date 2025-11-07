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

package scheduling

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"

	"k8s.io/kubernetes/pkg/apis/scheduling"
)

func TestWorkloadWarnings(t *testing.T) {
	testcases := []struct {
		name     string
		template *scheduling.Workload
		expected []string
	}{
		{
			name: "no warning",
			template: &scheduling.Workload{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Spec: scheduling.WorkloadSpec{
					PodGroups: []scheduling.PodGroup{
						{
							Name: "foo",
							Policy: scheduling.PodGroupPolicy{
								Gang: &scheduling.GangSchedulingPolicy{
									MinCount: 1,
								},
							},
						},
					},
				},
			},
			expected: nil,
		},
		{
			name: "warning",
			template: &scheduling.Workload{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Spec: scheduling.WorkloadSpec{
					PodGroups: []scheduling.PodGroup{
						{
							Name: "foo",
							Policy: scheduling.PodGroupPolicy{
								Gang: &scheduling.GangSchedulingPolicy{
									MinCount: 0,
								},
							},
						},
					},
				},
			},
			expected: []string{
				"podGroup.policy.gang.minCount: must be greater than 0",
			},
		},
	}

	for _, tc := range testcases {
		t.Run("workload_"+tc.name, func(t *testing.T) {
			actual := sets.New[string](GetWarningsForWorkload(tc.template)...)
			expected := sets.New[string](tc.expected...)
			for _, missing := range sets.List[string](expected.Difference(actual)) {
				t.Errorf("missing: %s", missing)
			}
			for _, extra := range sets.List[string](actual.Difference(expected)) {
				t.Errorf("extra: %s", extra)
			}
		})

	}
}
