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
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/golang/glog"

	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/kubernetes/pkg/api"
	apierrors "k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/apis/rbac"
	genericapirequest "k8s.io/kubernetes/pkg/genericapiserver/api/request"
	rbacregistryvalidation "k8s.io/kubernetes/pkg/registry/rbac/validation"
	"k8s.io/kubernetes/pkg/serviceaccount"
	utilerrors "k8s.io/kubernetes/pkg/util/errors"
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
		ruleResolver, _ := NewTestRuleResolver(tt.roles, tt.roleBindings, tt.clusterRoles, tt.clusterRoleBindings)
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

type AuthorizationRuleResolver interface {
	// GetRoleReferenceRules attempts to resolve the role reference of a RoleBinding or ClusterRoleBinding.  The passed namespace should be the namepsace
	// of the role binding, the empty string if a cluster role binding.
	GetRoleReferenceRules(roleRef rbac.RoleRef, namespace string) ([]rbac.PolicyRule, error)

	// RulesFor returns the list of rules that apply to a given user in a given namespace and error.  If an error is returned, the slice of
	// PolicyRules may not be complete, but it contains all retrievable rules.  This is done because policy rules are purely additive and policy determinations
	// can be made on the basis of those rules that are found.
	RulesFor(user user.Info, namespace string) ([]rbac.PolicyRule, error)
}

// ConfirmNoEscalation determines if the roles for a given user in a given namespace encompass the provided role.
func ConfirmNoEscalation(ctx genericapirequest.Context, ruleResolver AuthorizationRuleResolver, rules []rbac.PolicyRule) error {
	ruleResolutionErrors := []error{}

	user, ok := genericapirequest.UserFrom(ctx)
	if !ok {
		return fmt.Errorf("no user on context")
	}
	namespace, _ := genericapirequest.NamespaceFrom(ctx)

	ownerRules, err := ruleResolver.RulesFor(user, namespace)
	if err != nil {
		// As per AuthorizationRuleResolver contract, this may return a non fatal error with an incomplete list of policies. Log the error and continue.
		glog.V(1).Infof("non-fatal error getting local rules for %v: %v", user, err)
		ruleResolutionErrors = append(ruleResolutionErrors, err)
	}

	ownerRightsCover, missingRights := rbacregistryvalidation.Covers(ownerRules, rules)
	if !ownerRightsCover {
		user, _ := genericapirequest.UserFrom(ctx)
		return apierrors.NewUnauthorized(fmt.Sprintf("attempt to grant extra privileges: %v user=%v ownerrules=%v ruleResolutionErrors=%v", missingRights, user, ownerRules, ruleResolutionErrors))
	}
	return nil
}

type DefaultRuleResolver struct {
	roleGetter               RoleGetter
	roleBindingLister        RoleBindingLister
	clusterRoleGetter        ClusterRoleGetter
	clusterRoleBindingLister ClusterRoleBindingLister
}

func NewDefaultRuleResolver(roleGetter RoleGetter, roleBindingLister RoleBindingLister, clusterRoleGetter ClusterRoleGetter, clusterRoleBindingLister ClusterRoleBindingLister) *DefaultRuleResolver {
	return &DefaultRuleResolver{roleGetter, roleBindingLister, clusterRoleGetter, clusterRoleBindingLister}
}

type RoleGetter interface {
	GetRole(namespace, name string) (*rbac.Role, error)
}

type RoleBindingLister interface {
	ListRoleBindings(namespace string) ([]*rbac.RoleBinding, error)
}

type ClusterRoleGetter interface {
	GetClusterRole(name string) (*rbac.ClusterRole, error)
}

type ClusterRoleBindingLister interface {
	ListClusterRoleBindings() ([]*rbac.ClusterRoleBinding, error)
}

func (r *DefaultRuleResolver) RulesFor(user user.Info, namespace string) ([]rbac.PolicyRule, error) {
	policyRules := []rbac.PolicyRule{}
	errorlist := []error{}

	if clusterRoleBindings, err := r.clusterRoleBindingLister.ListClusterRoleBindings(); err != nil {
		errorlist = append(errorlist, err)

	} else {
		for _, clusterRoleBinding := range clusterRoleBindings {
			if !appliesTo(user, clusterRoleBinding.Subjects, "") {
				continue
			}
			rules, err := r.GetRoleReferenceRules(clusterRoleBinding.RoleRef, "")
			if err != nil {
				errorlist = append(errorlist, err)
				continue
			}
			policyRules = append(policyRules, rules...)
		}
	}

	if len(namespace) > 0 {
		if roleBindings, err := r.roleBindingLister.ListRoleBindings(namespace); err != nil {
			errorlist = append(errorlist, err)

		} else {
			for _, roleBinding := range roleBindings {
				if !appliesTo(user, roleBinding.Subjects, namespace) {
					continue
				}
				rules, err := r.GetRoleReferenceRules(roleBinding.RoleRef, namespace)
				if err != nil {
					errorlist = append(errorlist, err)
					continue
				}
				policyRules = append(policyRules, rules...)
			}
		}
	}

	return policyRules, utilerrors.NewAggregate(errorlist)
}

// GetRoleReferenceRules attempts to resolve the RoleBinding or ClusterRoleBinding.
func (r *DefaultRuleResolver) GetRoleReferenceRules(roleRef rbac.RoleRef, bindingNamespace string) ([]rbac.PolicyRule, error) {
	switch kind := rbac.RoleRefGroupKind(roleRef); kind {
	case rbac.Kind("Role"):
		role, err := r.roleGetter.GetRole(bindingNamespace, roleRef.Name)
		if err != nil {
			return nil, err
		}
		return role.Rules, nil

	case rbac.Kind("ClusterRole"):
		clusterRole, err := r.clusterRoleGetter.GetClusterRole(roleRef.Name)
		if err != nil {
			return nil, err
		}
		return clusterRole.Rules, nil

	default:
		return nil, fmt.Errorf("unsupported role reference kind: %q", kind)
	}
}
func appliesTo(user user.Info, bindingSubjects []rbac.Subject, namespace string) bool {
	for _, bindingSubject := range bindingSubjects {
		if appliesToUser(user, bindingSubject, namespace) {
			return true
		}
	}
	return false
}

func appliesToUser(user user.Info, subject rbac.Subject, namespace string) bool {
	switch subject.Kind {
	case rbac.UserKind:
		return user.GetName() == subject.Name

	case rbac.GroupKind:
		return has(user.GetGroups(), subject.Name)

	case rbac.ServiceAccountKind:
		// default the namespace to namespace we're working in if its available.  This allows rolebindings that reference
		// SAs in th local namespace to avoid having to qualify them.
		saNamespace := namespace
		if len(subject.Namespace) > 0 {
			saNamespace = subject.Namespace
		}
		if len(saNamespace) == 0 {
			return false
		}
		return serviceaccount.MakeUsername(saNamespace, subject.Name) == user.GetName()
	default:
		return false
	}
}

// NewTestRuleResolver returns a rule resolver from lists of role objects.
func NewTestRuleResolver(roles []*rbac.Role, roleBindings []*rbac.RoleBinding, clusterRoles []*rbac.ClusterRole, clusterRoleBindings []*rbac.ClusterRoleBinding) (AuthorizationRuleResolver, *StaticRoles) {
	r := StaticRoles{
		roles:               roles,
		roleBindings:        roleBindings,
		clusterRoles:        clusterRoles,
		clusterRoleBindings: clusterRoleBindings,
	}
	return newMockRuleResolver(&r), &r
}

func newMockRuleResolver(r *StaticRoles) AuthorizationRuleResolver {
	return NewDefaultRuleResolver(r, r, r, r)
}

// StaticRoles is a rule resolver that resolves from lists of role objects.
type StaticRoles struct {
	roles               []*rbac.Role
	roleBindings        []*rbac.RoleBinding
	clusterRoles        []*rbac.ClusterRole
	clusterRoleBindings []*rbac.ClusterRoleBinding
}

func (r *StaticRoles) GetRole(namespace, name string) (*rbac.Role, error) {
	if len(namespace) == 0 {
		return nil, errors.New("must provide namespace when getting role")
	}
	for _, role := range r.roles {
		if role.Namespace == namespace && role.Name == name {
			return role, nil
		}
	}
	return nil, errors.New("role not found")
}

func (r *StaticRoles) GetClusterRole(name string) (*rbac.ClusterRole, error) {
	for _, clusterRole := range r.clusterRoles {
		if clusterRole.Name == name {
			return clusterRole, nil
		}
	}
	return nil, errors.New("role not found")
}

func (r *StaticRoles) ListRoleBindings(namespace string) ([]*rbac.RoleBinding, error) {
	if len(namespace) == 0 {
		return nil, errors.New("must provide namespace when listing role bindings")
	}

	roleBindingList := []*rbac.RoleBinding{}
	for _, roleBinding := range r.roleBindings {
		if roleBinding.Namespace != namespace {
			continue
		}
		// TODO(ericchiang): need to implement label selectors?
		roleBindingList = append(roleBindingList, roleBinding)
	}
	return roleBindingList, nil
}

func (r *StaticRoles) ListClusterRoleBindings() ([]*rbac.ClusterRoleBinding, error) {
	return r.clusterRoleBindings, nil
}

func has(set []string, ele string) bool {
	for _, s := range set {
		if s == ele {
			return true
		}
	}
	return false
}
