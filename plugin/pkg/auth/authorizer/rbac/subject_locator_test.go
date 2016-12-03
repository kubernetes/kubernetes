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

package rbac

import (
	"reflect"
	"testing"

	"k8s.io/kubernetes/pkg/apis/rbac"
	"k8s.io/kubernetes/pkg/apis/rbac/validation"
	"k8s.io/kubernetes/pkg/auth/authorizer"
	"k8s.io/kubernetes/pkg/auth/user"
)

func TestSubjectLocator(t *testing.T) {
	type actionToSubjects struct {
		action   authorizer.Attributes
		subjects []rbac.Subject
	}

	tests := []struct {
		name                string
		roles               []*rbac.Role
		roleBindings        []*rbac.RoleBinding
		clusterRoles        []*rbac.ClusterRole
		clusterRoleBindings []*rbac.ClusterRoleBinding

		superUser string

		actionsToSubjects []actionToSubjects
	}{
		{
			name: "no super user, star matches star",
			clusterRoles: []*rbac.ClusterRole{
				newClusterRole("admin", newRule("*", "*", "*", "*")),
			},
			clusterRoleBindings: []*rbac.ClusterRoleBinding{
				newClusterRoleBinding("admin", "User:super-admin", "Group:super-admins"),
			},
			roleBindings: []*rbac.RoleBinding{
				newRoleBinding("ns1", "admin", bindToClusterRole, "User:admin", "Group:admins"),
			},
			actionsToSubjects: []actionToSubjects{
				{
					&defaultAttributes{"", "", "get", "Pods", "", "ns1", ""},
					[]rbac.Subject{
						{Kind: rbac.GroupKind, Name: user.SystemPrivilegedGroup},
						{Kind: rbac.UserKind, Name: "super-admin"},
						{Kind: rbac.GroupKind, Name: "super-admins"},
						{Kind: rbac.UserKind, Name: "admin"},
						{Kind: rbac.GroupKind, Name: "admins"},
					},
				},
				{
					// cluster role matches star in namespace
					&defaultAttributes{"", "", "*", "Pods", "", "*", ""},
					[]rbac.Subject{
						{Kind: rbac.GroupKind, Name: user.SystemPrivilegedGroup},
						{Kind: rbac.UserKind, Name: "super-admin"},
						{Kind: rbac.GroupKind, Name: "super-admins"},
					},
				},
				{
					// empty ns
					&defaultAttributes{"", "", "*", "Pods", "", "", ""},
					[]rbac.Subject{
						{Kind: rbac.GroupKind, Name: user.SystemPrivilegedGroup},
						{Kind: rbac.UserKind, Name: "super-admin"},
						{Kind: rbac.GroupKind, Name: "super-admins"},
					},
				},
			},
		},
		{
			name:      "super user, local roles work",
			superUser: "foo",
			clusterRoles: []*rbac.ClusterRole{
				newClusterRole("admin", newRule("*", "*", "*", "*")),
			},
			clusterRoleBindings: []*rbac.ClusterRoleBinding{
				newClusterRoleBinding("admin", "User:super-admin", "Group:super-admins"),
			},
			roles: []*rbac.Role{
				newRole("admin", "ns1", newRule("get", "*", "Pods", "*")),
			},
			roleBindings: []*rbac.RoleBinding{
				newRoleBinding("ns1", "admin", bindToRole, "User:admin", "Group:admins"),
			},
			actionsToSubjects: []actionToSubjects{
				{
					&defaultAttributes{"", "", "get", "Pods", "", "ns1", ""},
					[]rbac.Subject{
						{Kind: rbac.GroupKind, Name: user.SystemPrivilegedGroup},
						{Kind: rbac.UserKind, APIVersion: "v1alpha1", Name: "foo"},
						{Kind: rbac.UserKind, Name: "super-admin"},
						{Kind: rbac.GroupKind, Name: "super-admins"},
						{Kind: rbac.UserKind, Name: "admin"},
						{Kind: rbac.GroupKind, Name: "admins"},
					},
				},
				{
					// verb matchies correctly
					&defaultAttributes{"", "", "create", "Pods", "", "ns1", ""},
					[]rbac.Subject{
						{Kind: rbac.GroupKind, Name: user.SystemPrivilegedGroup},
						{Kind: rbac.UserKind, APIVersion: "v1alpha1", Name: "foo"},
						{Kind: rbac.UserKind, Name: "super-admin"},
						{Kind: rbac.GroupKind, Name: "super-admins"},
					},
				},
				{
					// binding only works in correct ns
					&defaultAttributes{"", "", "get", "Pods", "", "ns2", ""},
					[]rbac.Subject{
						{Kind: rbac.GroupKind, Name: user.SystemPrivilegedGroup},
						{Kind: rbac.UserKind, APIVersion: "v1alpha1", Name: "foo"},
						{Kind: rbac.UserKind, Name: "super-admin"},
						{Kind: rbac.GroupKind, Name: "super-admins"},
					},
				},
			},
		},
	}
	for _, tt := range tests {
		ruleResolver, lister := validation.NewTestRuleResolver(tt.roles, tt.roleBindings, tt.clusterRoles, tt.clusterRoleBindings)
		a := SubjectAccessEvaluator{tt.superUser, lister, lister, ruleResolver}
		for i, action := range tt.actionsToSubjects {
			actualSubjects, err := a.AllowedSubjects(action.action)
			if err != nil {
				t.Errorf("case %q %d: error %v", tt.name, i, err)
			}
			if !reflect.DeepEqual(actualSubjects, action.subjects) {
				t.Errorf("case %q %d: expected %v actual %v", tt.name, i, action.subjects, actualSubjects)
			}
		}
	}
}
