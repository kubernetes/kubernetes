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

package resource

import (
	"reflect"
	"testing"

	"k8s.io/apimachinery/pkg/runtime/schema"
)

func TestCategoryExpansion(t *testing.T) {
	tests := []struct {
		name string
		arg  string

		expected   []schema.GroupResource
		expectedOk bool
	}{
		{
			name:     "no-replacement",
			arg:      "service",
			expected: nil,
		},
		{
			name: "all-replacement",
			arg:  "all",
			expected: []schema.GroupResource{
				{Resource: "pods"},
				{Resource: "replicationcontrollers"},
				{Resource: "services"},
				{Resource: "statefulsets", Group: "apps"},
				{Resource: "horizontalpodautoscalers", Group: "autoscaling"},
				{Resource: "jobs", Group: "batch"},
				{Resource: "cronjobs", Group: "batch"},
				{Resource: "daemonsets", Group: "extensions"},
				{Resource: "deployments", Group: "extensions"},
				{Resource: "replicasets", Group: "extensions"},
			},
			expectedOk: true,
		},
	}

	for _, test := range tests {
		actual, actualOk := LegacyCategoryExpander.Expand(test.arg)
		if e, a := test.expected, actual; !reflect.DeepEqual(e, a) {
			t.Errorf("%s:  expected %s, got %s", test.name, e, a)
		}
		if e, a := test.expectedOk, actualOk; e != a {
			t.Errorf("%s:  expected %v, got %v", test.name, e, a)
		}
	}
}
