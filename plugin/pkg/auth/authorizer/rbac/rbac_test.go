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
	"context"
	"fmt"
	"strings"
	"testing"

	rbacv1 "k8s.io/api/rbac/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	rbacv1helpers "k8s.io/kubernetes/pkg/apis/rbac/v1"
	rbacregistryvalidation "k8s.io/kubernetes/pkg/registry/rbac/validation"
	"k8s.io/kubernetes/plugin/pkg/auth/authorizer/rbac/bootstrappolicy"
)

func newRule(verbs, apiGroups, resources, nonResourceURLs string) rbacv1.PolicyRule {
	return rbacv1.PolicyRule{
		Verbs:           strings.Split(verbs, ","),
		APIGroups:       strings.Split(apiGroups, ","),
		Resources:       strings.Split(resources, ","),
		NonResourceURLs: strings.Split(nonResourceURLs, ","),
	}
}

func newRole(name, namespace string, rules ...rbacv1.PolicyRule) *rbacv1.Role {
	return &rbacv1.Role{ObjectMeta: metav1.ObjectMeta{Namespace: namespace, Name: name}, Rules: rules}
}

func newClusterRole(name string, rules ...rbacv1.PolicyRule) *rbacv1.ClusterRole {
	return &rbacv1.ClusterRole{ObjectMeta: metav1.ObjectMeta{Name: name}, Rules: rules}
}

const (
	bindToRole        uint16 = 0x0
	bindToClusterRole uint16 = 0x1
)

func newClusterRoleBinding(roleName string, subjects ...string) *rbacv1.ClusterRoleBinding {
	r := &rbacv1.ClusterRoleBinding{
		ObjectMeta: metav1.ObjectMeta{},
		RoleRef: rbacv1.RoleRef{
			APIGroup: rbacv1.GroupName,
			Kind:     "ClusterRole", // ClusterRoleBindings can only refer to ClusterRole
			Name:     roleName,
		},
	}

	r.Subjects = make([]rbacv1.Subject, len(subjects))
	for i, subject := range subjects {
		split := strings.SplitN(subject, ":", 2)
		r.Subjects[i].Kind, r.Subjects[i].Name = split[0], split[1]

		switch r.Subjects[i].Kind {
		case rbacv1.ServiceAccountKind:
			r.Subjects[i].APIGroup = ""
		case rbacv1.UserKind, rbacv1.GroupKind:
			r.Subjects[i].APIGroup = rbacv1.GroupName
		default:
			panic(fmt.Errorf("invalid kind %s", r.Subjects[i].Kind))
		}
	}
	return r
}

func newRoleBinding(namespace, roleName string, bindType uint16, subjects ...string) *rbacv1.RoleBinding {
	r := &rbacv1.RoleBinding{ObjectMeta: metav1.ObjectMeta{Namespace: namespace}}

	switch bindType {
	case bindToRole:
		r.RoleRef = rbacv1.RoleRef{APIGroup: rbacv1.GroupName, Kind: "Role", Name: roleName}
	case bindToClusterRole:
		r.RoleRef = rbacv1.RoleRef{APIGroup: rbacv1.GroupName, Kind: "ClusterRole", Name: roleName}
	}

	r.Subjects = make([]rbacv1.Subject, len(subjects))
	for i, subject := range subjects {
		split := strings.SplitN(subject, ":", 2)
		r.Subjects[i].Kind, r.Subjects[i].Name = split[0], split[1]

		switch r.Subjects[i].Kind {
		case rbacv1.ServiceAccountKind:
			r.Subjects[i].APIGroup = ""
		case rbacv1.UserKind, rbacv1.GroupKind:
			r.Subjects[i].APIGroup = rbacv1.GroupName
		default:
			panic(fmt.Errorf("invalid kind %s", r.Subjects[i].Kind))
		}
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
func (d *defaultAttributes) ParseFieldSelector() (fields.Selector, error) {
	panic("not supported for RBAC")
}
func (d *defaultAttributes) ParseLabelSelector() (labels.Selector, error) {
	panic("not supported for RBAC")
}

func TestAuthorizer(t *testing.T) {
	tests := []struct {
		roles               []*rbacv1.Role
		roleBindings        []*rbacv1.RoleBinding
		clusterRoles        []*rbacv1.ClusterRole
		clusterRoleBindings []*rbacv1.ClusterRoleBinding

		shouldPass []authorizer.Attributes
		shouldFail []authorizer.Attributes
	}{
		{
			clusterRoles: []*rbacv1.ClusterRole{
				newClusterRole("admin", newRule("*", "*", "*", "*")),
			},
			roleBindings: []*rbacv1.RoleBinding{
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
			clusterRoles: []*rbacv1.ClusterRole{
				newClusterRole("non-resource-url-getter", newRule("get", "", "", "/apis")),
				newClusterRole("non-resource-url", newRule("*", "", "", "/apis")),
				newClusterRole("non-resource-url-prefix", newRule("get", "", "", "/apis/*")),
			},
			clusterRoleBindings: []*rbacv1.ClusterRoleBinding{
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
			clusterRoles: []*rbacv1.ClusterRole{
				newClusterRole("admin", newRule("*", "*", "pods", "*")),
			},
			roleBindings: []*rbacv1.RoleBinding{
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
			clusterRoles: []*rbacv1.ClusterRole{
				newClusterRole("admin",
					newRule("*", "*", "pods/status", "*"),
					newRule("*", "*", "*/scale", "*"),
				),
			},
			roleBindings: []*rbacv1.RoleBinding{
				newRoleBinding("ns1", "admin", bindToClusterRole, "User:admin", "Group:admins"),
			},
			shouldPass: []authorizer.Attributes{
				&defaultAttributes{"admin", "", "get", "pods", "status", "ns1", ""},
				&defaultAttributes{"admin", "", "get", "pods", "scale", "ns1", ""},
				&defaultAttributes{"admin", "", "get", "deployments", "scale", "ns1", ""},
				&defaultAttributes{"admin", "", "get", "anything", "scale", "ns1", ""},
			},
			shouldFail: []authorizer.Attributes{
				&defaultAttributes{"admin", "", "get", "pods", "", "ns1", ""},
			},
		},
	}
	for i, tt := range tests {
		ruleResolver, _ := rbacregistryvalidation.NewTestRuleResolver(tt.roles, tt.roleBindings, tt.clusterRoles, tt.clusterRoleBindings)
		a := RBACAuthorizer{ruleResolver}
		for _, attr := range tt.shouldPass {
			if decision, _, _ := a.Authorize(context.Background(), attr); decision != authorizer.DecisionAllow {
				t.Errorf("case %d: incorrectly restricted %s", i, attr)
			}
		}

		for _, attr := range tt.shouldFail {
			if decision, _, _ := a.Authorize(context.Background(), attr); decision == authorizer.DecisionAllow {
				t.Errorf("case %d: incorrectly passed %s", i, attr)
			}
		}
	}
}

func TestRuleMatches(t *testing.T) {
	type requestToTest struct {
		request  authorizer.AttributesRecord
		expected bool
	}
	tests := []struct {
		name string
		rule rbacv1.PolicyRule

		requestsToExpected []requestToTest
	}{
		{
			name: "star verb, exact match other",
			rule: rbacv1helpers.NewRule("*").Groups("group1").Resources("resource1").RuleOrDie(),
			requestsToExpected: []requestToTest{
				{resourceRequest("verb1").Group("group1").Resource("resource1").New(), true},
				{resourceRequest("verb1").Group("group2").Resource("resource1").New(), false},
				{resourceRequest("verb1").Group("group1").Resource("resource2").New(), false},
				{resourceRequest("verb1").Group("group2").Resource("resource2").New(), false},
				{resourceRequest("verb2").Group("group1").Resource("resource1").New(), true},
				{resourceRequest("verb2").Group("group2").Resource("resource1").New(), false},
				{resourceRequest("verb2").Group("group1").Resource("resource2").New(), false},
				{resourceRequest("verb2").Group("group2").Resource("resource2").New(), false},
			},
		},
		{
			name: "star group, exact match other",
			rule: rbacv1helpers.NewRule("verb1").Groups("*").Resources("resource1").RuleOrDie(),
			requestsToExpected: []requestToTest{
				{resourceRequest("verb1").Group("group1").Resource("resource1").New(), true},
				{resourceRequest("verb1").Group("group2").Resource("resource1").New(), true},
				{resourceRequest("verb1").Group("group1").Resource("resource2").New(), false},
				{resourceRequest("verb1").Group("group2").Resource("resource2").New(), false},
				{resourceRequest("verb2").Group("group1").Resource("resource1").New(), false},
				{resourceRequest("verb2").Group("group2").Resource("resource1").New(), false},
				{resourceRequest("verb2").Group("group1").Resource("resource2").New(), false},
				{resourceRequest("verb2").Group("group2").Resource("resource2").New(), false},
			},
		},
		{
			name: "star resource, exact match other",
			rule: rbacv1helpers.NewRule("verb1").Groups("group1").Resources("*").RuleOrDie(),
			requestsToExpected: []requestToTest{
				{resourceRequest("verb1").Group("group1").Resource("resource1").New(), true},
				{resourceRequest("verb1").Group("group2").Resource("resource1").New(), false},
				{resourceRequest("verb1").Group("group1").Resource("resource2").New(), true},
				{resourceRequest("verb1").Group("group2").Resource("resource2").New(), false},
				{resourceRequest("verb2").Group("group1").Resource("resource1").New(), false},
				{resourceRequest("verb2").Group("group2").Resource("resource1").New(), false},
				{resourceRequest("verb2").Group("group1").Resource("resource2").New(), false},
				{resourceRequest("verb2").Group("group2").Resource("resource2").New(), false},
			},
		},
		{
			name: "tuple expansion",
			rule: rbacv1helpers.NewRule("verb1", "verb2").Groups("group1", "group2").Resources("resource1", "resource2").RuleOrDie(),
			requestsToExpected: []requestToTest{
				{resourceRequest("verb1").Group("group1").Resource("resource1").New(), true},
				{resourceRequest("verb1").Group("group2").Resource("resource1").New(), true},
				{resourceRequest("verb1").Group("group1").Resource("resource2").New(), true},
				{resourceRequest("verb1").Group("group2").Resource("resource2").New(), true},
				{resourceRequest("verb2").Group("group1").Resource("resource1").New(), true},
				{resourceRequest("verb2").Group("group2").Resource("resource1").New(), true},
				{resourceRequest("verb2").Group("group1").Resource("resource2").New(), true},
				{resourceRequest("verb2").Group("group2").Resource("resource2").New(), true},
			},
		},
		{
			name: "subresource expansion",
			rule: rbacv1helpers.NewRule("*").Groups("*").Resources("resource1/subresource1").RuleOrDie(),
			requestsToExpected: []requestToTest{
				{resourceRequest("verb1").Group("group1").Resource("resource1").Subresource("subresource1").New(), true},
				{resourceRequest("verb1").Group("group2").Resource("resource1").Subresource("subresource2").New(), false},
				{resourceRequest("verb1").Group("group1").Resource("resource2").Subresource("subresource1").New(), false},
				{resourceRequest("verb1").Group("group2").Resource("resource2").Subresource("subresource1").New(), false},
				{resourceRequest("verb2").Group("group1").Resource("resource1").Subresource("subresource1").New(), true},
				{resourceRequest("verb2").Group("group2").Resource("resource1").Subresource("subresource2").New(), false},
				{resourceRequest("verb2").Group("group1").Resource("resource2").Subresource("subresource1").New(), false},
				{resourceRequest("verb2").Group("group2").Resource("resource2").Subresource("subresource1").New(), false},
			},
		},
		{
			name: "star nonresource, exact match other",
			rule: rbacv1helpers.NewRule("verb1").URLs("*").RuleOrDie(),
			requestsToExpected: []requestToTest{
				{nonresourceRequest("verb1").URL("/foo").New(), true},
				{nonresourceRequest("verb1").URL("/foo/bar").New(), true},
				{nonresourceRequest("verb1").URL("/foo/baz").New(), true},
				{nonresourceRequest("verb1").URL("/foo/bar/one").New(), true},
				{nonresourceRequest("verb1").URL("/foo/baz/one").New(), true},
				{nonresourceRequest("verb2").URL("/foo").New(), false},
				{nonresourceRequest("verb2").URL("/foo/bar").New(), false},
				{nonresourceRequest("verb2").URL("/foo/baz").New(), false},
				{nonresourceRequest("verb2").URL("/foo/bar/one").New(), false},
				{nonresourceRequest("verb2").URL("/foo/baz/one").New(), false},
			},
		},
		{
			name: "star nonresource subpath",
			rule: rbacv1helpers.NewRule("verb1").URLs("/foo/*").RuleOrDie(),
			requestsToExpected: []requestToTest{
				{nonresourceRequest("verb1").URL("/foo").New(), false},
				{nonresourceRequest("verb1").URL("/foo/bar").New(), true},
				{nonresourceRequest("verb1").URL("/foo/baz").New(), true},
				{nonresourceRequest("verb1").URL("/foo/bar/one").New(), true},
				{nonresourceRequest("verb1").URL("/foo/baz/one").New(), true},
				{nonresourceRequest("verb1").URL("/notfoo").New(), false},
				{nonresourceRequest("verb1").URL("/notfoo/bar").New(), false},
				{nonresourceRequest("verb1").URL("/notfoo/baz").New(), false},
				{nonresourceRequest("verb1").URL("/notfoo/bar/one").New(), false},
				{nonresourceRequest("verb1").URL("/notfoo/baz/one").New(), false},
			},
		},
		{
			name: "star verb, exact nonresource",
			rule: rbacv1helpers.NewRule("*").URLs("/foo", "/foo/bar/one").RuleOrDie(),
			requestsToExpected: []requestToTest{
				{nonresourceRequest("verb1").URL("/foo").New(), true},
				{nonresourceRequest("verb1").URL("/foo/bar").New(), false},
				{nonresourceRequest("verb1").URL("/foo/baz").New(), false},
				{nonresourceRequest("verb1").URL("/foo/bar/one").New(), true},
				{nonresourceRequest("verb1").URL("/foo/baz/one").New(), false},
				{nonresourceRequest("verb2").URL("/foo").New(), true},
				{nonresourceRequest("verb2").URL("/foo/bar").New(), false},
				{nonresourceRequest("verb2").URL("/foo/baz").New(), false},
				{nonresourceRequest("verb2").URL("/foo/bar/one").New(), true},
				{nonresourceRequest("verb2").URL("/foo/baz/one").New(), false},
			},
		},
	}
	for _, tc := range tests {
		for _, requestToTest := range tc.requestsToExpected {
			if e, a := requestToTest.expected, RuleAllows(requestToTest.request, &tc.rule); e != a {
				t.Errorf("%q: expected %v, got %v for %v", tc.name, e, a, requestToTest.request)
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

func BenchmarkAuthorize(b *testing.B) {
	bootstrapRoles := []rbacv1.ClusterRole{}
	bootstrapRoles = append(bootstrapRoles, bootstrappolicy.ControllerRoles()...)
	bootstrapRoles = append(bootstrapRoles, bootstrappolicy.ClusterRoles()...)

	bootstrapBindings := []rbacv1.ClusterRoleBinding{}
	bootstrapBindings = append(bootstrapBindings, bootstrappolicy.ClusterRoleBindings()...)
	bootstrapBindings = append(bootstrapBindings, bootstrappolicy.ControllerRoleBindings()...)

	clusterRoles := []*rbacv1.ClusterRole{}
	for i := range bootstrapRoles {
		clusterRoles = append(clusterRoles, &bootstrapRoles[i])
	}
	clusterRoleBindings := []*rbacv1.ClusterRoleBinding{}
	for i := range bootstrapBindings {
		clusterRoleBindings = append(clusterRoleBindings, &bootstrapBindings[i])
	}

	_, resolver := rbacregistryvalidation.NewTestRuleResolver(nil, nil, clusterRoles, clusterRoleBindings)

	authz := New(resolver, resolver, resolver, resolver)

	nodeUser := &user.DefaultInfo{Name: "system:node:node1", Groups: []string{"system:nodes", "system:authenticated"}}
	requests := []struct {
		name  string
		attrs authorizer.Attributes
	}{
		{
			"allow list pods",
			authorizer.AttributesRecord{
				ResourceRequest: true,
				User:            nodeUser,
				Verb:            "list",
				Resource:        "pods",
				Subresource:     "",
				Name:            "",
				Namespace:       "",
				APIGroup:        "",
				APIVersion:      "v1",
			},
		},
		{
			"allow update pods/status",
			authorizer.AttributesRecord{
				ResourceRequest: true,
				User:            nodeUser,
				Verb:            "update",
				Resource:        "pods",
				Subresource:     "status",
				Name:            "mypods",
				Namespace:       "myns",
				APIGroup:        "",
				APIVersion:      "v1",
			},
		},
		{
			"forbid educate dolphins",
			authorizer.AttributesRecord{
				ResourceRequest: true,
				User:            nodeUser,
				Verb:            "educate",
				Resource:        "dolphins",
				Subresource:     "",
				Name:            "",
				Namespace:       "",
				APIGroup:        "",
				APIVersion:      "v1",
			},
		},
	}

	b.ResetTimer()
	for _, request := range requests {
		b.Run(request.name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				authz.Authorize(context.Background(), request.attrs)
			}
		})
	}
}
