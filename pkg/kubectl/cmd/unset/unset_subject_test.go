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

package unset

import (
	"reflect"
	"testing"

	"k8s.io/kubernetes/pkg/apis/rbac"
)

func TestRemoveSubject(t *testing.T) {
	tests := []struct {
		Name       string
		existing   []rbac.Subject
		subjects   []rbac.Subject
		expected   []rbac.Subject
		wantChange bool
	}{
		{
			Name: "remove user which is not exist",
			existing: []rbac.Subject{
				{
					APIGroup: "rbac.authorization.k8s.io",
					Kind:     "User",
					Name:     "a",
				},
				{
					APIGroup: "rbac.authorization.k8s.io",
					Kind:     "User",
					Name:     "b",
				},
			},
			subjects: []rbac.Subject{
				{
					APIGroup: "rbac.authorization.k8s.io",
					Kind:     "User",
					Name:     "c",
				},
			},
			expected: []rbac.Subject{
				{
					APIGroup: "rbac.authorization.k8s.io",
					Kind:     "User",
					Name:     "a",
				},
				{
					APIGroup: "rbac.authorization.k8s.io",
					Kind:     "User",
					Name:     "b",
				},
			},
			wantChange: false,
		},
		{
			Name: "remove serviceaccounts which is exist",
			existing: []rbac.Subject{
				{
					Kind:      "ServiceAccount",
					Namespace: "one",
					Name:      "a",
				},
				{
					Kind:      "ServiceAccount",
					Namespace: "one",
					Name:      "b",
				},
			},
			subjects: []rbac.Subject{
				{
					Kind:      "ServiceAccount",
					Namespace: "one",
					Name:      "a",
				},
			},
			expected: []rbac.Subject{
				{
					Kind:      "ServiceAccount",
					Namespace: "one",
					Name:      "b",
				},
			},
			wantChange: true,
		},
	}
	for _, tt := range tests {
		changed := false
		got := []rbac.Subject{}
		if changed, got = removeSubjects(tt.existing, tt.subjects); (changed != false) != tt.wantChange {
			t.Errorf("%q. removeSubjects() changed = %v, wantChange = %v", tt.Name, changed, tt.wantChange)
		}

		want := tt.expected
		if !reflect.DeepEqual(got, want) {
			t.Errorf("%q. removeSubjects() failed", tt.Name)
			t.Errorf("Got: %v", got)
			t.Errorf("Want: %v", want)
		}
	}
}
