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
	"fmt"
	"strings"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/rbac"
	"k8s.io/kubernetes/pkg/apis/rbac/validation"
	"k8s.io/kubernetes/pkg/auth/authorizer"
	"k8s.io/kubernetes/pkg/auth/user"
)

func newRule(verbs, apiGroups, resources, nonResourceURLs string) rbac.PolicyRule {
	return rbac.PolicyRule{
		Verbs:           strings.Split(verbs, ","),
		APIGroups:       strings.Split(apiGroups, ","),
		Resources:       strings.Split(resources, ","),
		NonResourceURLs: strings.Split(nonResourceURLs, ","),
	}
}

func newRole(name, namespace string, rules ...rbac.PolicyRule) *rbac.Role {
	return &rbac.Role{ObjectMeta: api.ObjectMeta{Namespace: namespace, Name: name}, Rules: rules}
}

func newClusterRole(name string, rules ...rbac.PolicyRule) *rbac.ClusterRole {
	return &rbac.ClusterRole{ObjectMeta: api.ObjectMeta{Name: name}, Rules: rules}
}

const (
	bindToRole        uint16 = 0x0
	bindToClusterRole uint16 = 0x1
)

func newClusterRoleBinding(roleName string, subjects ...string) *rbac.ClusterRoleBinding {
	r := &rbac.ClusterRoleBinding{
		ObjectMeta: api.ObjectMeta{},
		RoleRef: rbac.RoleRef{
			APIGroup: rbac.GroupName,
			Kind:     "ClusterRole", // ClusterRoleBindings can only refer to ClusterRole
			Name:     roleName,
		},
	}

	r.Subjects = make([]rbac.Subject, len(subjects))
	for i, subject := range subjects {
		split := strings.SplitN(subject, ":", 2)
		r.Subjects[i].Kind, r.Subjects[i].Name = split[0], split[1]
	}
	return r
}

func newRoleBinding(namespace, roleName string, bindType uint16, subjects ...string) *rbac.RoleBinding {
	r := &rbac.RoleBinding{ObjectMeta: api.ObjectMeta{Namespace: namespace}}

	switch bindType {
	case bindToRole:
		r.RoleRef = rbac.RoleRef{APIGroup: rbac.GroupName, Kind: "Role", Name: roleName}
	case bindToClusterRole:
		r.RoleRef = rbac.RoleRef{APIGroup: rbac.GroupName, Kind: "ClusterRole", Name: roleName}
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

func (d *defaultAttributes) GetUser() user.Info {
	return &user.DefaultInfo{Name: d.user, Groups: strings.Split(d.groups, ",")}
}
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
		roles               []*rbac.Role
		roleBindings        []*rbac.RoleBinding
		clusterRoles        []*rbac.ClusterRole
		clusterRoleBindings []*rbac.ClusterRoleBinding

		shouldPass []authorizer.Attributes
		shouldFail []authorizer.Attributes
	}{
		{
			clusterRoles: []*rbac.ClusterRole{
				newClusterRole("admin", newRule("*", "*", "*", "*")),
			},
			roleBindings: []*rbac.RoleBinding{
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
			// Non-resource-url tests
			clusterRoles: []*rbac.ClusterRole{
				newClusterRole("non-resource-url-getter", newRule("get", "", "", "/apis")),
				newClusterRole("non-resource-url", newRule("*", "", "", "/apis")),
				newClusterRole("non-resource-url-prefix", newRule("get", "", "", "/apis/*")),
			},
			clusterRoleBindings: []*rbac.ClusterRoleBinding{
				newClusterRoleBinding("non-resource-url-getter", "User:foo", "Group:bar"),
				newClusterRoleBinding("non-resource-url", "User:admin", "Group:admin"),
				newClusterRoleBinding("non-resource-url-prefix", "User:prefixed", "Group:prefixed"),
			},
			shouldPass: []authorizer.Attributes{
				authorizer.AttributesRecord{User: &user.DefaultInfo{Name: "foo"}, Verb: "get", Path: "/apis"},
				authorizer.AttributesRecord{User: &user.DefaultInfo{Groups: []string{"bar"}}, Verb: "get", Path: "/apis"},
				authorizer.AttributesRecord{User: &user.DefaultInfo{Name: "admin"}, Verb: "get", Path: "/apis"},
				authorizer.AttributesRecord{User: &user.DefaultInfo{Groups: []string{"admin"}}, Verb: "get", Path: "/apis"},
				authorizer.AttributesRecord{User: &user.DefaultInfo{Name: "admin"}, Verb: "watch", Path: "/apis"},
				authorizer.AttributesRecord{User: &user.DefaultInfo{Groups: []string{"admin"}}, Verb: "watch", Path: "/apis"},

				authorizer.AttributesRecord{User: &user.DefaultInfo{Name: "prefixed"}, Verb: "get", Path: "/apis/v1"},
				authorizer.AttributesRecord{User: &user.DefaultInfo{Groups: []string{"prefixed"}}, Verb: "get", Path: "/apis/v1"},
				authorizer.AttributesRecord{User: &user.DefaultInfo{Name: "prefixed"}, Verb: "get", Path: "/apis/v1/foobar"},
				authorizer.AttributesRecord{User: &user.DefaultInfo{Groups: []string{"prefixed"}}, Verb: "get", Path: "/apis/v1/foorbar"},
			},
			shouldFail: []authorizer.Attributes{
				// wrong verb
				authorizer.AttributesRecord{User: &user.DefaultInfo{Name: "foo"}, Verb: "watch", Path: "/apis"},
				authorizer.AttributesRecord{User: &user.DefaultInfo{Groups: []string{"bar"}}, Verb: "watch", Path: "/apis"},

				// wrong path
				authorizer.AttributesRecord{User: &user.DefaultInfo{Name: "foo"}, Verb: "get", Path: "/api/v1"},
				authorizer.AttributesRecord{User: &user.DefaultInfo{Groups: []string{"bar"}}, Verb: "get", Path: "/api/v1"},
				authorizer.AttributesRecord{User: &user.DefaultInfo{Name: "admin"}, Verb: "get", Path: "/api/v1"},
				authorizer.AttributesRecord{User: &user.DefaultInfo{Groups: []string{"admin"}}, Verb: "get", Path: "/api/v1"},

				// not covered by prefix
				authorizer.AttributesRecord{User: &user.DefaultInfo{Name: "prefixed"}, Verb: "get", Path: "/api/v1"},
				authorizer.AttributesRecord{User: &user.DefaultInfo{Groups: []string{"prefixed"}}, Verb: "get", Path: "/api/v1"},
			},
		},
		{
			// test subresource resolution
			clusterRoles: []*rbac.ClusterRole{
				newClusterRole("admin", newRule("*", "*", "pods", "*")),
			},
			roleBindings: []*rbac.RoleBinding{
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
			clusterRoles: []*rbac.ClusterRole{
				newClusterRole("admin", newRule("*", "*", "pods/status", "*")),
			},
			roleBindings: []*rbac.RoleBinding{
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
		ruleResolver, _ := validation.NewTestRuleResolver(tt.roles, tt.roleBindings, tt.clusterRoles, tt.clusterRoleBindings)
		a := RBACAuthorizer{ruleResolver}
		for _, attr := range tt.shouldPass {
			if authorized, _, _ := a.Authorize(attr); !authorized {
				t.Errorf("case %d: incorrectly restricted %s", i, attr)
			}
		}

		for _, attr := range tt.shouldFail {
			if authorized, _, _ := a.Authorize(attr); authorized {
				t.Errorf("case %d: incorrectly passed %s", i, attr)
			}
		}
	}
}

func TestRuleMatches(t *testing.T) {
	tests := []struct {
		name string
		rule rbac.PolicyRule

		requestsToExpected map[authorizer.AttributesRecord]bool
	}{
		{
			name: "star verb, exact match other",
			rule: rbac.NewRule("*").Groups("group1").Resources("resource1").RuleOrDie(),
			requestsToExpected: map[authorizer.AttributesRecord]bool{
				resourceRequest("verb1").Group("group1").Resource("resource1").New(): true,
				resourceRequest("verb1").Group("group2").Resource("resource1").New(): false,
				resourceRequest("verb1").Group("group1").Resource("resource2").New(): false,
				resourceRequest("verb1").Group("group2").Resource("resource2").New(): false,
				resourceRequest("verb2").Group("group1").Resource("resource1").New(): true,
				resourceRequest("verb2").Group("group2").Resource("resource1").New(): false,
				resourceRequest("verb2").Group("group1").Resource("resource2").New(): false,
				resourceRequest("verb2").Group("group2").Resource("resource2").New(): false,
			},
		},
		{
			name: "star group, exact match other",
			rule: rbac.NewRule("verb1").Groups("*").Resources("resource1").RuleOrDie(),
			requestsToExpected: map[authorizer.AttributesRecord]bool{
				resourceRequest("verb1").Group("group1").Resource("resource1").New(): true,
				resourceRequest("verb1").Group("group2").Resource("resource1").New(): true,
				resourceRequest("verb1").Group("group1").Resource("resource2").New(): false,
				resourceRequest("verb1").Group("group2").Resource("resource2").New(): false,
				resourceRequest("verb2").Group("group1").Resource("resource1").New(): false,
				resourceRequest("verb2").Group("group2").Resource("resource1").New(): false,
				resourceRequest("verb2").Group("group1").Resource("resource2").New(): false,
				resourceRequest("verb2").Group("group2").Resource("resource2").New(): false,
			},
		},
		{
			name: "star resource, exact match other",
			rule: rbac.NewRule("verb1").Groups("group1").Resources("*").RuleOrDie(),
			requestsToExpected: map[authorizer.AttributesRecord]bool{
				resourceRequest("verb1").Group("group1").Resource("resource1").New(): true,
				resourceRequest("verb1").Group("group2").Resource("resource1").New(): false,
				resourceRequest("verb1").Group("group1").Resource("resource2").New(): true,
				resourceRequest("verb1").Group("group2").Resource("resource2").New(): false,
				resourceRequest("verb2").Group("group1").Resource("resource1").New(): false,
				resourceRequest("verb2").Group("group2").Resource("resource1").New(): false,
				resourceRequest("verb2").Group("group1").Resource("resource2").New(): false,
				resourceRequest("verb2").Group("group2").Resource("resource2").New(): false,
			},
		},
		{
			name: "tuple expansion",
			rule: rbac.NewRule("verb1", "verb2").Groups("group1", "group2").Resources("resource1", "resource2").RuleOrDie(),
			requestsToExpected: map[authorizer.AttributesRecord]bool{
				resourceRequest("verb1").Group("group1").Resource("resource1").New(): true,
				resourceRequest("verb1").Group("group2").Resource("resource1").New(): true,
				resourceRequest("verb1").Group("group1").Resource("resource2").New(): true,
				resourceRequest("verb1").Group("group2").Resource("resource2").New(): true,
				resourceRequest("verb2").Group("group1").Resource("resource1").New(): true,
				resourceRequest("verb2").Group("group2").Resource("resource1").New(): true,
				resourceRequest("verb2").Group("group1").Resource("resource2").New(): true,
				resourceRequest("verb2").Group("group2").Resource("resource2").New(): true,
			},
		},
		{
			name: "subresource expansion",
			rule: rbac.NewRule("*").Groups("*").Resources("resource1/subresource1").RuleOrDie(),
			requestsToExpected: map[authorizer.AttributesRecord]bool{
				resourceRequest("verb1").Group("group1").Resource("resource1").Subresource("subresource1").New(): true,
				resourceRequest("verb1").Group("group2").Resource("resource1").Subresource("subresource2").New(): false,
				resourceRequest("verb1").Group("group1").Resource("resource2").Subresource("subresource1").New(): false,
				resourceRequest("verb1").Group("group2").Resource("resource2").Subresource("subresource1").New(): false,
				resourceRequest("verb2").Group("group1").Resource("resource1").Subresource("subresource1").New(): true,
				resourceRequest("verb2").Group("group2").Resource("resource1").Subresource("subresource2").New(): false,
				resourceRequest("verb2").Group("group1").Resource("resource2").Subresource("subresource1").New(): false,
				resourceRequest("verb2").Group("group2").Resource("resource2").Subresource("subresource1").New(): false,
			},
		},
		{
			name: "star nonresource, exact match other",
			rule: rbac.NewRule("verb1").URLs("*").RuleOrDie(),
			requestsToExpected: map[authorizer.AttributesRecord]bool{
				nonresourceRequest("verb1").URL("/foo").New():         true,
				nonresourceRequest("verb1").URL("/foo/bar").New():     true,
				nonresourceRequest("verb1").URL("/foo/baz").New():     true,
				nonresourceRequest("verb1").URL("/foo/bar/one").New(): true,
				nonresourceRequest("verb1").URL("/foo/baz/one").New(): true,
				nonresourceRequest("verb2").URL("/foo").New():         false,
				nonresourceRequest("verb2").URL("/foo/bar").New():     false,
				nonresourceRequest("verb2").URL("/foo/baz").New():     false,
				nonresourceRequest("verb2").URL("/foo/bar/one").New(): false,
				nonresourceRequest("verb2").URL("/foo/baz/one").New(): false,
			},
		},
		{
			name: "star nonresource subpath",
			rule: rbac.NewRule("verb1").URLs("/foo/*").RuleOrDie(),
			requestsToExpected: map[authorizer.AttributesRecord]bool{
				nonresourceRequest("verb1").URL("/foo").New():            false,
				nonresourceRequest("verb1").URL("/foo/bar").New():        true,
				nonresourceRequest("verb1").URL("/foo/baz").New():        true,
				nonresourceRequest("verb1").URL("/foo/bar/one").New():    true,
				nonresourceRequest("verb1").URL("/foo/baz/one").New():    true,
				nonresourceRequest("verb1").URL("/notfoo").New():         false,
				nonresourceRequest("verb1").URL("/notfoo/bar").New():     false,
				nonresourceRequest("verb1").URL("/notfoo/baz").New():     false,
				nonresourceRequest("verb1").URL("/notfoo/bar/one").New(): false,
				nonresourceRequest("verb1").URL("/notfoo/baz/one").New(): false,
			},
		},
		{
			name: "star verb, exact nonresource",
			rule: rbac.NewRule("*").URLs("/foo", "/foo/bar/one").RuleOrDie(),
			requestsToExpected: map[authorizer.AttributesRecord]bool{
				nonresourceRequest("verb1").URL("/foo").New():         true,
				nonresourceRequest("verb1").URL("/foo/bar").New():     false,
				nonresourceRequest("verb1").URL("/foo/baz").New():     false,
				nonresourceRequest("verb1").URL("/foo/bar/one").New(): true,
				nonresourceRequest("verb1").URL("/foo/baz/one").New(): false,
				nonresourceRequest("verb2").URL("/foo").New():         true,
				nonresourceRequest("verb2").URL("/foo/bar").New():     false,
				nonresourceRequest("verb2").URL("/foo/baz").New():     false,
				nonresourceRequest("verb2").URL("/foo/bar/one").New(): true,
				nonresourceRequest("verb2").URL("/foo/baz/one").New(): false,
			},
		},
	}
	for _, tc := range tests {
		for request, expected := range tc.requestsToExpected {
			if e, a := expected, RuleAllows(request, tc.rule); e != a {
				t.Errorf("%q: expected %v, got %v for %v", tc.name, e, a, request)
			}
		}
	}
}

type requestAttributeBuilder struct {
	request authorizer.AttributesRecord
}

func resourceRequest(verb string) *requestAttributeBuilder {
	return &requestAttributeBuilder{
		request: authorizer.AttributesRecord{ResourceRequest: true, Verb: verb},
	}
}

func nonresourceRequest(verb string) *requestAttributeBuilder {
	return &requestAttributeBuilder{
		request: authorizer.AttributesRecord{ResourceRequest: false, Verb: verb},
	}
}

func (r *requestAttributeBuilder) Group(group string) *requestAttributeBuilder {
	r.request.APIGroup = group
	return r
}

func (r *requestAttributeBuilder) Resource(resource string) *requestAttributeBuilder {
	r.request.Resource = resource
	return r
}

func (r *requestAttributeBuilder) Subresource(subresource string) *requestAttributeBuilder {
	r.request.Subresource = subresource
	return r
}

func (r *requestAttributeBuilder) Name(name string) *requestAttributeBuilder {
	r.request.Name = name
	return r
}

func (r *requestAttributeBuilder) URL(url string) *requestAttributeBuilder {
	r.request.Path = url
	return r
}

func (r *requestAttributeBuilder) New() authorizer.AttributesRecord {
	return r.request
}
