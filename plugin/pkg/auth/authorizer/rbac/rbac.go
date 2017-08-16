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

// Package rbac implements the authorizer.Authorizer interface using roles base access control.
package rbac

import (
	"fmt"

	"github.com/golang/glog"

	"bytes"

	"k8s.io/apimachinery/pkg/labels"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/kubernetes/pkg/apis/rbac"
	rbaclisters "k8s.io/kubernetes/pkg/client/listers/rbac/internalversion"
	rbacregistryvalidation "k8s.io/kubernetes/pkg/registry/rbac/validation"
)

type RequestToRuleMapper interface {
	// RulesFor returns all known PolicyRules and any errors that happened while locating those rules.
	// Any rule returned is still valid, since rules are deny by default.  If you can pass with the rules
	// supplied, you do not have to fail the request.  If you cannot, you should indicate the error along
	// with your denial.
	RulesFor(subject user.Info, namespace string) ([]rbac.PolicyRule, error)

	// VisitRulesFor invokes visitor() with each rule that applies to a given user in a given namespace,
	// and each error encountered resolving those rules. Rule may be nil if err is non-nil.
	// If visitor() returns false, visiting is short-circuited.
	VisitRulesFor(user user.Info, namespace string, visitor func(rule *rbac.PolicyRule, err error) bool)
}

type RBACAuthorizer struct {
	authorizationRuleResolver RequestToRuleMapper
}

// authorizingVisitor short-circuits once allowed, and collects any resolution errors encountered
type authorizingVisitor struct {
	requestAttributes authorizer.Attributes

	allowed bool
	errors  []error
}

func (v *authorizingVisitor) visit(rule *rbac.PolicyRule, err error) bool {
	if rule != nil && RuleAllows(v.requestAttributes, rule) {
		v.allowed = true
		return false
	}
	if err != nil {
		v.errors = append(v.errors, err)
	}
	return true
}

func (r *RBACAuthorizer) Authorize(requestAttributes authorizer.Attributes) (bool, string, error) {
	ruleCheckingVisitor := &authorizingVisitor{requestAttributes: requestAttributes}

	r.authorizationRuleResolver.VisitRulesFor(requestAttributes.GetUser(), requestAttributes.GetNamespace(), ruleCheckingVisitor.visit)
	if ruleCheckingVisitor.allowed {
		return true, "", nil
	}

	// Build a detailed log of the denial.
	// Make the whole block conditional so we don't do a lot of string-building we won't use.
	if glog.V(2) {
		var operation string
		if requestAttributes.IsResourceRequest() {
			b := &bytes.Buffer{}
			b.WriteString(`"`)
			b.WriteString(requestAttributes.GetVerb())
			b.WriteString(`" resource "`)
			b.WriteString(requestAttributes.GetResource())
			if len(requestAttributes.GetAPIGroup()) > 0 {
				b.WriteString(`.`)
				b.WriteString(requestAttributes.GetAPIGroup())
			}
			if len(requestAttributes.GetSubresource()) > 0 {
				b.WriteString(`/`)
				b.WriteString(requestAttributes.GetSubresource())
			}
			b.WriteString(`"`)
			if len(requestAttributes.GetName()) > 0 {
				b.WriteString(` named "`)
				b.WriteString(requestAttributes.GetName())
				b.WriteString(`"`)
			}
			operation = b.String()
		} else {
			operation = fmt.Sprintf("%q nonResourceURL %q", requestAttributes.GetVerb(), requestAttributes.GetPath())
		}

		var scope string
		if ns := requestAttributes.GetNamespace(); len(ns) > 0 {
			scope = fmt.Sprintf("in namespace %q", ns)
		} else {
			scope = "cluster-wide"
		}

		glog.Infof("RBAC DENY: user %q groups %q cannot %s %s", requestAttributes.GetUser().GetName(), requestAttributes.GetUser().GetGroups(), operation, scope)
	}

	reason := ""
	if len(ruleCheckingVisitor.errors) > 0 {
		reason = fmt.Sprintf("%v", utilerrors.NewAggregate(ruleCheckingVisitor.errors))
	}
	return false, reason, nil
}

func New(roles rbacregistryvalidation.RoleGetter, roleBindings rbacregistryvalidation.RoleBindingLister, clusterRoles rbacregistryvalidation.ClusterRoleGetter, clusterRoleBindings rbacregistryvalidation.ClusterRoleBindingLister) *RBACAuthorizer {
	authorizer := &RBACAuthorizer{
		authorizationRuleResolver: rbacregistryvalidation.NewDefaultRuleResolver(
			roles, roleBindings, clusterRoles, clusterRoleBindings,
		),
	}
	return authorizer
}

func RulesAllow(requestAttributes authorizer.Attributes, rules ...rbac.PolicyRule) bool {
	for i := range rules {
		if RuleAllows(requestAttributes, &rules[i]) {
			return true
		}
	}

	return false
}

func RuleAllows(requestAttributes authorizer.Attributes, rule *rbac.PolicyRule) bool {
	if requestAttributes.IsResourceRequest() {
		resource := requestAttributes.GetResource()
		if len(requestAttributes.GetSubresource()) > 0 {
			resource = requestAttributes.GetResource() + "/" + requestAttributes.GetSubresource()
		}

		return rbac.VerbMatches(rule, requestAttributes.GetVerb()) &&
			rbac.APIGroupMatches(rule, requestAttributes.GetAPIGroup()) &&
			rbac.ResourceMatches(rule, resource) &&
			rbac.ResourceNameMatches(rule, requestAttributes.GetName())
	}

	return rbac.VerbMatches(rule, requestAttributes.GetVerb()) &&
		rbac.NonResourceURLMatches(rule, requestAttributes.GetPath())
}

type RoleGetter struct {
	Lister rbaclisters.RoleLister
}

func (g *RoleGetter) GetRole(namespace, name string) (*rbac.Role, error) {
	return g.Lister.Roles(namespace).Get(name)
}

type RoleBindingLister struct {
	Lister rbaclisters.RoleBindingLister
}

func (l *RoleBindingLister) ListRoleBindings(namespace string) ([]*rbac.RoleBinding, error) {
	return l.Lister.RoleBindings(namespace).List(labels.Everything())
}

type ClusterRoleGetter struct {
	Lister rbaclisters.ClusterRoleLister
}

func (g *ClusterRoleGetter) GetClusterRole(name string) (*rbac.ClusterRole, error) {
	return g.Lister.Get(name)
}

type ClusterRoleBindingLister struct {
	Lister rbaclisters.ClusterRoleBindingLister
}

func (l *ClusterRoleBindingLister) ListClusterRoleBindings() ([]*rbac.ClusterRoleBinding, error) {
	return l.Lister.List(labels.Everything())
}
