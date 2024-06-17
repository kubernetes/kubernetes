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

package create

import (
	"strconv"
	"testing"

	rbac "k8s.io/api/rbac/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestCreateClusterRoleBinding(t *testing.T) {
	tests := []struct {
		options  *ClusterRoleBindingOptions
		expected *rbac.ClusterRoleBinding
	}{
		{
			options: &ClusterRoleBindingOptions{
				ClusterRole:     "fake-clusterrole",
				Users:           []string{"fake-user"},
				Groups:          []string{"fake-group"},
				ServiceAccounts: []string{"fake-namespace:fake-account"},
				Name:            "fake-binding",
			},
			expected: &rbac.ClusterRoleBinding{
				TypeMeta: v1.TypeMeta{
					Kind:       "ClusterRoleBinding",
					APIVersion: "rbac.authorization.k8s.io/v1",
				},
				ObjectMeta: v1.ObjectMeta{
					Name: "fake-binding",
				},
				RoleRef: rbac.RoleRef{
					APIGroup: rbac.GroupName,
					Kind:     "ClusterRole",
					Name:     "fake-clusterrole",
				},
				Subjects: []rbac.Subject{
					{
						Kind:     rbac.UserKind,
						APIGroup: "rbac.authorization.k8s.io",
						Name:     "fake-user",
					},
					{
						Kind:     rbac.GroupKind,
						APIGroup: "rbac.authorization.k8s.io",
						Name:     "fake-group",
					},
					{
						Kind:      rbac.ServiceAccountKind,
						Namespace: "fake-namespace",
						Name:      "fake-account",
					},
				},
			},
		},
	}

	for i, tc := range tests {
		t.Run(strconv.Itoa(i), func(t *testing.T) {
			clusterRoleBinding, err := tc.options.createClusterRoleBinding()
			if err != nil {
				t.Errorf("unexpected error:\n%#v\n", err)
				return
			}
			if !apiequality.Semantic.DeepEqual(clusterRoleBinding, tc.expected) {
				t.Errorf("expected:\n%#v\ngot:\n%#v", tc.expected, clusterRoleBinding)
			}
		})
	}

}
