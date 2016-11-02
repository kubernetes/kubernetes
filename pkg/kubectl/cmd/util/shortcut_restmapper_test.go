/*
Copyright 2016 The Kubernetes Authors.

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

package util

import (
	"strings"
	"testing"

	"k8s.io/kubernetes/pkg/api/testapi"
)

func TestReplaceAliases(t *testing.T) {
	tests := []struct {
		name     string
		arg      string
		expected string
	}{
		{
			name:     "no-replacement",
			arg:      "service",
			expected: "service",
		},
		{
			name:     "all-replacement",
			arg:      "all",
			expected: "pods,replicationcontrollers,services,statefulsets,horizontalpodautoscalers,jobs,deployments,replicasets",
		},
		{
			name:     "alias-in-comma-separated-arg",
			arg:      "all,secrets",
			expected: "pods,replicationcontrollers,services,statefulsets,horizontalpodautoscalers,jobs,deployments,replicasets,secrets",
		},
	}

	mapper := NewShortcutExpander(testapi.Default.RESTMapper(), nil)

	for _, test := range tests {
		resources := []string{}
		for _, arg := range strings.Split(test.arg, ",") {
			curr, _ := mapper.AliasesForResource(arg)
			resources = append(resources, curr...)
		}
		if strings.Join(resources, ",") != test.expected {
			t.Errorf("%s: unexpected argument: expected %s, got %s", test.name, test.expected, resources)
		}
	}
}
