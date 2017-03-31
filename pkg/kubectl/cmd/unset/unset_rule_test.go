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

func TestRemovePolicyRuleForObject(t *testing.T) {
	tests := []struct {
		name     string
		existing []rbac.PolicyRule
		rule     *rbac.PolicyRule
		expected []rbac.PolicyRule
		wantErr  bool
	}{
		{
			name: "remove resource with same verbs in role",
			existing: []rbac.PolicyRule{
				{
					APIGroups: []string{""},
					Resources: []string{"pods", "namespaces"},
					Verbs:     []string{"delete"},
				},
			},
			rule: &rbac.PolicyRule{
				APIGroups: []string{""},
				Resources: []string{"pods"},
				Verbs:     []string{"delete"},
			},
			expected: []rbac.PolicyRule{
				{
					APIGroups: []string{""},
					Resources: []string{"namespaces"},
					Verbs:     []string{"delete"},
				},
			},
			wantErr: false,
		},
		{
			name: "remove resource with same resources in role",
			existing: []rbac.PolicyRule{
				{
					APIGroups: []string{""},
					Resources: []string{"pods", "namespaces"},
					Verbs:     []string{"delete", "get"},
				},
			},
			rule: &rbac.PolicyRule{
				APIGroups: []string{""},
				Resources: []string{"pods", "namespaces"},
				Verbs:     []string{"delete"},
			},
			expected: []rbac.PolicyRule{
				{
					APIGroups: []string{""},
					Resources: []string{"pods", "namespaces"},
					Verbs:     []string{"get"},
				},
			},
			wantErr: false,
		},
		{
			name: "remove resource with same resource name in role",
			existing: []rbac.PolicyRule{
				{
					APIGroups:     []string{""},
					Resources:     []string{"pods", "namespaces"},
					Verbs:         []string{"delete", "get"},
					ResourceNames: []string{"foo"},
				},
			},
			rule: &rbac.PolicyRule{
				APIGroups:     []string{""},
				Resources:     []string{"pods", "namespaces"},
				Verbs:         []string{"delete"},
				ResourceNames: []string{"foo"},
			},
			expected: []rbac.PolicyRule{
				{
					APIGroups:     []string{""},
					Resources:     []string{"pods", "namespaces"},
					Verbs:         []string{"get"},
					ResourceNames: []string{"foo"},
				},
			},
			wantErr: false,
		},
		{
			name: "remove resource not exist in role",
			existing: []rbac.PolicyRule{
				{
					APIGroups: []string{""},
					Resources: []string{"pods", "namespaces"},
					Verbs:     []string{"delete", "get"},
				},
			},
			rule: &rbac.PolicyRule{
				APIGroups: []string{""},
				Resources: []string{"pods", "namespaces"},
				Verbs:     []string{"list"},
			},
			expected: []rbac.PolicyRule{
				{
					APIGroups: []string{""},
					Resources: []string{"pods", "namespaces"},
					Verbs:     []string{"delete", "get"},
				},
			},
			wantErr: false,
		},
		{
			name: "remove resource with mutli resource names in role",
			existing: []rbac.PolicyRule{
				{
					APIGroups:     []string{""},
					Resources:     []string{"pods"},
					Verbs:         []string{"delete", "get"},
					ResourceNames: []string{"foo"},
				},
				{
					APIGroups:     []string{""},
					Resources:     []string{"pods"},
					Verbs:         []string{"delete", "get"},
					ResourceNames: []string{"bar"},
				},
			},
			rule: &rbac.PolicyRule{
				APIGroups:     []string{""},
				Resources:     []string{"pods"},
				Verbs:         []string{"delete", "get"},
				ResourceNames: []string{"foo"},
			},
			expected: []rbac.PolicyRule{
				{
					APIGroups:     []string{""},
					Resources:     []string{"pods"},
					Verbs:         []string{"delete", "get"},
					ResourceNames: []string{"bar"},
				},
			},
			wantErr: false,
		},
		{
			name: "remove resource with wildcard permission in role",
			existing: []rbac.PolicyRule{
				{
					APIGroups: []string{""},
					Resources: []string{"pods", "namespaces"},
					Verbs:     []string{"*"},
				},
			},
			rule: &rbac.PolicyRule{
				APIGroups: []string{""},
				Resources: []string{"pods"},
				Verbs:     []string{"create", "get", "delete", "list", "watch"},
			},
			wantErr: true,
		},
		{
			name: "remove wildcard permission resource from role",
			existing: []rbac.PolicyRule{
				{
					APIGroups: []string{""},
					Resources: []string{"pods", "namespaces"},
					Verbs:     []string{"delete", "get"},
				},
			},
			rule: &rbac.PolicyRule{
				APIGroups: []string{""},
				Resources: []string{"pods"},
				Verbs:     []string{"*"},
			},
			wantErr: true,
		},
		{
			name: "remove wildcard permission resource from role which have wildcard",
			existing: []rbac.PolicyRule{
				{
					APIGroups: []string{""},
					Resources: []string{"pods", "namespaces"},
					Verbs:     []string{"*"},
				},
			},
			rule: &rbac.PolicyRule{
				APIGroups: []string{""},
				Resources: []string{"pods"},
				Verbs:     []string{"*"},
			},
			expected: []rbac.PolicyRule{
				{
					APIGroups: []string{""},
					Resources: []string{"namespaces"},
					Verbs:     []string{"*"},
				},
			},
			wantErr: false,
		},
		{
			name: "remove resource in clusterrole",
			existing: []rbac.PolicyRule{
				{
					APIGroups: []string{""},
					Resources: []string{"pods", "namespaces"},
					Verbs:     []string{"delete", "get"},
				},
			},
			rule: &rbac.PolicyRule{
				APIGroups: []string{""},
				Resources: []string{"pods"},
				Verbs:     []string{"delete", "get"},
			},
			expected: []rbac.PolicyRule{
				{
					APIGroups: []string{""},
					Resources: []string{"namespaces"},
					Verbs:     []string{"delete", "get"},
				},
			},
			wantErr: false,
		},
		{
			name: "remove non resource in clusterrole",
			existing: []rbac.PolicyRule{
				{
					NonResourceURLs: []string{"/apis/*", "/apis/v1"},
					Verbs:           []string{"delete"},
				},
			},
			rule: &rbac.PolicyRule{
				NonResourceURLs: []string{"/apis/*"},
				Verbs:           []string{"delete"},
			},
			expected: []rbac.PolicyRule{
				{
					NonResourceURLs: []string{"/apis/v1"},
					Verbs:           []string{"delete"},
				},
			},
			wantErr: false,
		},
		{
			name: "remove non resource with wildcard permission in clusterrole",
			existing: []rbac.PolicyRule{
				{
					NonResourceURLs: []string{"/apis/*", "/apis/v1"},
					Verbs:           []string{"*"},
				},
			},
			rule: &rbac.PolicyRule{
				NonResourceURLs: []string{"/apis/*"},
				Verbs:           []string{"get", "post"},
			},
			wantErr: true,
		},
		{
			name: "remove wildcard permission non rerource from clusterrole",
			existing: []rbac.PolicyRule{
				{
					NonResourceURLs: []string{"/apis/*", "/apis/v1"},
					Verbs:           []string{"delete"},
				},
			},
			rule: &rbac.PolicyRule{
				NonResourceURLs: []string{"/apis/*"},
				Verbs:           []string{"*"},
			},
			wantErr: true,
		},
		{
			name: "remove wildcard permission non rerource from clusterrole which have wildcard",
			existing: []rbac.PolicyRule{
				{
					NonResourceURLs: []string{"/apis/*", "/apis/v1"},
					Verbs:           []string{"*"},
				},
			},
			rule: &rbac.PolicyRule{
				NonResourceURLs: []string{"/apis/*"},
				Verbs:           []string{"*"},
			},
			expected: []rbac.PolicyRule{
				{
					NonResourceURLs: []string{"/apis/v1"},
					Verbs:           []string{"*"},
				},
			},
			wantErr: false,
		},
	}
	for _, tt := range tests {
		var err error
		got := []rbac.PolicyRule{}
		if got, err = removePolicyRule(tt.existing, tt.rule); (err != nil) != tt.wantErr {
			t.Errorf("%q. removePolicyRule() error = %v, wantErr %v", tt.name, err, tt.wantErr)
		}

		want := tt.expected
		if !reflect.DeepEqual(got, want) {
			t.Errorf("%q. removePolicyRule() failed", tt.name)
			t.Errorf("Got: %v", got)
			t.Errorf("Want: %v", want)
		}
	}
}
