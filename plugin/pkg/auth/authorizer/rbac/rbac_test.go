/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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
	"fmt"
	"strings"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/rbac"
	"k8s.io/kubernetes/pkg/apis/rbac/validation"
	"k8s.io/kubernetes/pkg/auth/authorizer"
)

func newRule(verbs, apiGroups, resources string) rbac.PolicyRule {
	return rbac.PolicyRule{
		Verbs:     strings.Split(verbs, ","),
		APIGroups: strings.Split(apiGroups, ","),
		Resources: strings.Split(resources, ","),
	}
}

func newRole(name, namespace string, rules ...rbac.PolicyRule) rbac.Role {
	return rbac.Role{ObjectMeta: api.ObjectMeta{Namespace: namespace, Name: name}, Rules: rules}
}

func newClusterRole(name string, rules ...rbac.PolicyRule) rbac.ClusterRole {
	return rbac.ClusterRole{ObjectMeta: api.ObjectMeta{Name: name}, Rules: rules}
}

const (
	bindToRole        uint16 = 0x0
	bindToClusterRole uint16 = 0x1
)

func newRoleBinding(namespace, roleName string, bindType uint16, subjects ...string) rbac.RoleBinding {
	r := rbac.RoleBinding{ObjectMeta: api.ObjectMeta{Namespace: namespace}}

	switch bindType {
	case bindToRole:
		r.RoleRef = api.ObjectReference{Kind: "Role", Namespace: namespace, Name: roleName}
	case bindToClusterRole:
		r.RoleRef = api.ObjectReference{Kind: "ClusterRole", Name: roleName}
	}

	r.Subjects = make([]rbac.Subject, len(subjects))
	for i, subject := range subjects {
		split := strings.SplitN(subject, ":", 2)
		r.Subjects[i].Kind, r.Subjects[i].Name = split[0], split[1]
	}
	return r
}

type defaultAttributes struct {
	user        string
	groups      string
	verb        string
	resource    string
	subresource string
	namespace   string
	apiGroup    string
}

func (d *defaultAttributes) String() string {
	return fmt.Sprintf("user=(%s), groups=(%s), verb=(%s), resource=(%s), namespace=(%s), apiGroup=(%s)",
		d.user, strings.Split(d.groups, ","), d.verb, d.resource, d.namespace, d.apiGroup)
}

func (d *defaultAttributes) GetUserName() string     { return d.user }
func (d *defaultAttributes) GetGroups() []string     { return strings.Split(d.groups, ",") }
func (d *defaultAttributes) GetVerb() string         { return d.verb }
func (d *defaultAttributes) IsReadOnly() bool        { return d.verb == "get" || d.verb == "watch" }
func (d *defaultAttributes) GetNamespace() string    { return d.namespace }
func (d *defaultAttributes) GetResource() string     { return d.resource }
func (d *defaultAttributes) GetSubresource() string  { return d.subresource }
func (d *defaultAttributes) GetName() string         { return "" }
func (d *defaultAttributes) GetAPIGroup() string     { return d.apiGroup }
func (d *defaultAttributes) GetAPIVersion() string   { return "" }
func (d *defaultAttributes) IsResourceRequest() bool { return true }
func (d *defaultAttributes) GetPath() string         { return "" }

func TestAuthorizer(t *testing.T) {
	tests := []struct {
		roles               []rbac.Role
		roleBindings        []rbac.RoleBinding
		clusterRoles        []rbac.ClusterRole
		clusterRoleBindings []rbac.ClusterRoleBinding

		superUser string

		shouldPass []authorizer.Attributes
		shouldFail []authorizer.Attributes
	}{
		{
			clusterRoles: []rbac.ClusterRole{
				newClusterRole("admin", newRule("*", "*", "*")),
			},
			roleBindings: []rbac.RoleBinding{
				newRoleBinding("ns1", "admin", bindToClusterRole, "User:admin", "Group:admins"),
			},
			shouldPass: []authorizer.Attributes{
				&defaultAttributes{"admin", "", "get", "Pods", "", "ns1", ""},
				&defaultAttributes{"admin", "", "watch", "Pods", "", "ns1", ""},
				&defaultAttributes{"admin", "group1", "watch", "Foobar", "", "ns1", ""},
				&defaultAttributes{"joe", "admins", "watch", "Foobar", "", "ns1", ""},
				&defaultAttributes{"joe", "group1,admins", "watch", "Foobar", "", "ns1", ""},
			},
			shouldFail: []authorizer.Attributes{
				&defaultAttributes{"admin", "", "GET", "Pods", "", "ns2", ""},
				&defaultAttributes{"admin", "", "GET", "Nodes", "", "", ""},
				&defaultAttributes{"admin", "admins", "GET", "Pods", "", "ns2", ""},
				&defaultAttributes{"admin", "admins", "GET", "Nodes", "", "", ""},
			},
		},
		{
			// test subresource resolution
			clusterRoles: []rbac.ClusterRole{
				newClusterRole("admin", newRule("*", "*", "pods")),
			},
			roleBindings: []rbac.RoleBinding{
				newRoleBinding("ns1", "admin", bindToClusterRole, "User:admin", "Group:admins"),
			},
			shouldPass: []authorizer.Attributes{
				&defaultAttributes{"admin", "", "get", "pods", "", "ns1", ""},
			},
			shouldFail: []authorizer.Attributes{
				&defaultAttributes{"admin", "", "get", "pods", "status", "ns1", ""},
			},
		},
		{
			// test subresource resolution
			clusterRoles: []rbac.ClusterRole{
				newClusterRole("admin", newRule("*", "*", "pods/status")),
			},
			roleBindings: []rbac.RoleBinding{
				newRoleBinding("ns1", "admin", bindToClusterRole, "User:admin", "Group:admins"),
			},
			shouldPass: []authorizer.Attributes{
				&defaultAttributes{"admin", "", "get", "pods", "status", "ns1", ""},
			},
			shouldFail: []authorizer.Attributes{
				&defaultAttributes{"admin", "", "get", "pods", "", "ns1", ""},
			},
		},
	}
	for i, tt := range tests {
		ruleResolver := validation.NewTestRuleResolver(tt.roles, tt.roleBindings, tt.clusterRoles, tt.clusterRoleBindings)
		a := RBACAuthorizer{tt.superUser, ruleResolver}
		for _, attr := range tt.shouldPass {
			if err := a.Authorize(attr); err != nil {
				t.Errorf("case %d: incorrectly restricted %s: %T %v", i, attr, err, err)
			}
		}

		for _, attr := range tt.shouldFail {
			if err := a.Authorize(attr); err == nil {
				t.Errorf("case %d: incorrectly passed %s", i, attr)
			}
		}
	}
}
