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

	rbacv1 "k8s.io/api/rbac/v1"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"

	rbacregistryvalidation "k8s.io/kubernetes/pkg/registry/rbac/validation"
)

func TestSubjectLocator(t *testing.T) {
	type actionToSubjects struct {
		action   authorizer.Attributes
		subjects []rbacv1.Subject
	}

	tests := []struct {
		name                string
		roles               []*rbacv1.Role
		roleBindings        []*rbacv1.RoleBinding
		clusterRoles        []*rbacv1.ClusterRole
		clusterRoleBindings []*rbacv1.ClusterRoleBinding

		superUser string

		actionsToSubjects []actionToSubjects
	}{
		{
			name: "no super user, star matches star",
			clusterRoles: []*rbacv1.ClusterRole{
				newClusterRole("admin", newRule("*", "*", "*", "*")),
			},
			clusterRoleBindings: []*rbacv1.ClusterRoleBinding{
				newClusterRoleBinding("admin", "User:super-admin", "Group:super-admins"),
			},
			roleBindings: []*rbacv1.RoleBinding{
				newRoleBinding("ns1", "admin", bindToClusterRole, "User:admin", "Group:admins"),
			},
			actionsToSubjects: []actionToSubjects{
				{
					&defaultAttributes{"", "", "get", "Pods", "", "ns1", ""},
					[]rbacv1.Subject{
						{Kind: rbacv1.GroupKind, APIGroup: rbacv1.GroupName, Name: user.SystemPrivilegedGroup},
						{Kind: rbacv1.UserKind, APIGroup: rbacv1.GroupName, Name: "super-admin"},
						{Kind: rbacv1.GroupKind, APIGroup: rbacv1.GroupName, Name: "super-admins"},
						{Kind: rbacv1.UserKind, APIGroup: rbacv1.GroupName, Name: "admin"},
						{Kind: rbacv1.GroupKind, APIGroup: rbacv1.GroupName, Name: "admins"},
					},
				},
				{
					// cluster role matches star in namespace
					&defaultAttributes{"", "", "*", "Pods", "", "*", ""},
					[]rbacv1.Subject{
						{Kind: rbacv1.GroupKind, APIGroup: rbacv1.GroupName, Name: user.SystemPrivilegedGroup},
						{Kind: rbacv1.UserKind, APIGroup: rbacv1.GroupName, Name: "super-admin"},
						{Kind: rbacv1.GroupKind, APIGroup: rbacv1.GroupName, Name: "super-admins"},
					},
				},
				{
					// empty ns
					&defaultAttributes{"", "", "*", "Pods", "", "", ""},
					[]rbacv1.Subject{
						{Kind: rbacv1.GroupKind, APIGroup: rbacv1.GroupName, Name: user.SystemPrivilegedGroup},
						{Kind: rbacv1.UserKind, APIGroup: rbacv1.GroupName, Name: "super-admin"},
						{Kind: rbacv1.GroupKind, APIGroup: rbacv1.GroupName, Name: "super-admins"},
					},
				},
			},
		},
		{
			name:      "super user, local roles work",
			superUser: "foo",
			clusterRoles: []*rbacv1.ClusterRole{
				newClusterRole("admin", newRule("*", "*", "*", "*")),
			},
			clusterRoleBindings: []*rbacv1.ClusterRoleBinding{
				newClusterRoleBinding("admin", "User:super-admin", "Group:super-admins"),
			},
			roles: []*rbacv1.Role{
				newRole("admin", "ns1", newRule("get", "*", "Pods", "*")),
			},
			roleBindings: []*rbacv1.RoleBinding{
				newRoleBinding("ns1", "admin", bindToRole, "User:admin", "Group:admins"),
			},
			actionsToSubjects: []actionToSubjects{
				{
					&defaultAttributes{"", "", "get", "Pods", "", "ns1", ""},
					[]rbacv1.Subject{
						{Kind: rbacv1.GroupKind, APIGroup: rbacv1.GroupName, Name: user.SystemPrivilegedGroup},
						{Kind: rbacv1.UserKind, APIGroup: rbacv1.GroupName, Name: "foo"},
						{Kind: rbacv1.UserKind, APIGroup: rbacv1.GroupName, Name: "super-admin"},
						{Kind: rbacv1.GroupKind, APIGroup: rbacv1.GroupName, Name: "super-admins"},
						{Kind: rbacv1.UserKind, APIGroup: rbacv1.GroupName, Name: "admin"},
						{Kind: rbacv1.GroupKind, APIGroup: rbacv1.GroupName, Name: "admins"},
					},
				},
				{
					// verb matchies correctly
					&defaultAttributes{"", "", "create", "Pods", "", "ns1", ""},
					[]rbacv1.Subject{
						{Kind: rbacv1.GroupKind, APIGroup: rbacv1.GroupName, Name: user.SystemPrivilegedGroup},
						{Kind: rbacv1.UserKind, APIGroup: rbacv1.GroupName, Name: "foo"},
						{Kind: rbacv1.UserKind, APIGroup: rbacv1.GroupName, Name: "super-admin"},
						{Kind: rbacv1.GroupKind, APIGroup: rbacv1.GroupName, Name: "super-admins"},
					},
				},
				{
					// binding only works in correct ns
					&defaultAttributes{"", "", "get", "Pods", "", "ns2", ""},
					[]rbacv1.Subject{
						{Kind: rbacv1.GroupKind, APIGroup: rbacv1.GroupName, Name: user.SystemPrivilegedGroup},
						{Kind: rbacv1.UserKind, APIGroup: rbacv1.GroupName, Name: "foo"},
						{Kind: rbacv1.UserKind, APIGroup: rbacv1.GroupName, Name: "super-admin"},
						{Kind: rbacv1.GroupKind, APIGroup: rbacv1.GroupName, Name: "super-admins"},
					},
				},
			},
		},
	}
	for _, tt := range tests {
		ruleResolver, lister := rbacregistryvalidation.NewTestRuleResolver(tt.roles, tt.roleBindings, tt.clusterRoles, tt.clusterRoleBindings)
		a := SubjectAccessEvaluator{tt.superUser, lister, lister, ruleResolver}
		for i, action := range tt.actionsToSubjects {
			actualSubjects, err := a.AllowedSubjects(genericapirequest.NewContext(), action.action)
			if err != nil {
				t.Errorf("case %q %d: error %v", tt.name, i, err)
			}
			if !reflect.DeepEqual(actualSubjects, action.subjects) {
				t.Errorf("case %q %d: expected\n%v\nactual\n%v", tt.name, i, action.subjects, actualSubjects)
			}
		}
	}
}
