/*
Copyright 2020 The Kubernetes Authors.

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

package v1

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/sets"
)

func TestGetNodeRoles(t *testing.T) {
	testCases := []struct {
		name   string
		labels map[string]string
		expect []string
	}{
		{
			name:   "no label",
			labels: nil,
			expect: []string{},
		},
		{
			name: "no role",
			labels: map[string]string{
				"foo": "bar",
			},
			expect: []string{},
		},
		{
			name: "old role",
			labels: map[string]string{
				"foo":                    "bar",
				LabelLegacyNodeLabelRole: "old-role",
			},
			expect: []string{"old-role"},
		},
		{
			name: "new roles - one role",
			labels: map[string]string{
				"foo":                         "bar",
				LabelNodeRolePrefix + "role1": "role1",
			},
			expect: []string{"role1"},
		},
		{
			name: "new roles - two roles",
			labels: map[string]string{
				"foo":                         "bar",
				LabelNodeRolePrefix + "role1": "role1",
				LabelNodeRolePrefix + "role2": "role2",
			},
			expect: []string{"role1", "role2"},
		},
		{
			name: "old role and new roles",
			labels: map[string]string{
				"foo":                         "bar",
				LabelLegacyNodeLabelRole:      "old-role",
				LabelNodeRolePrefix + "role1": "role1",
				LabelNodeRolePrefix + "role2": "role2",
			},
			expect: []string{"old-role", "role1", "role2"},
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			result := GetNodeRoles(testCase.labels)
			if !sets.NewString(testCase.expect...).Equal(sets.NewString(result...)) {
				t.Errorf("expected %v, got %v", testCase.expect, result)
			}
		})
	}
}
