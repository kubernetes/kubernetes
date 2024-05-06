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

package node

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"

	"k8s.io/kubernetes/pkg/apis/node"
)

func TestWarnings(t *testing.T) {
	testcases := []struct {
		name     string
		template *node.RuntimeClass
		expected []string
	}{
		{
			name:     "null",
			template: nil,
			expected: nil,
		},
		{
			name: "no warning",
			template: &node.RuntimeClass{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
			},
			expected: nil,
		},
		{
			name: "warning",
			template: &node.RuntimeClass{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Scheduling: &node.Scheduling{
					NodeSelector: map[string]string{
						"beta.kubernetes.io/arch": "amd64",
						"beta.kubernetes.io/os":   "linux",
					},
				},
			},
			expected: []string{
				`scheduling.nodeSelector: deprecated since v1.14; use "kubernetes.io/arch" instead`,
				`scheduling.nodeSelector: deprecated since v1.14; use "kubernetes.io/os" instead`,
			},
		},
	}

	for _, tc := range testcases {
		t.Run("podspec_"+tc.name, func(t *testing.T) {
			actual := sets.New[string](GetWarningsForRuntimeClass(tc.template)...)
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
