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

	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/kubernetes/pkg/apis/rbac"
	rbacregistryvalidation "k8s.io/kubernetes/pkg/registry/rbac/validation"
)

type RequestToRuleMapper interface {
	// RulesFor returns all known PolicyRules and any errors that happened while locating those rules.
	// Any rule returned is still valid, since rules are deny by default.  If you can pass with the rules
	// supplied, you do not have to fail the request.  If you cannot, you should indicate the error along
	// with your denial.
	RulesFor(subject user.Info, namespace string) ([]rbac.PolicyRule, error)
}

type RBACAuthorizer struct {
	authorizationRuleResolver RequestToRuleMapper
}

func (r *RBACAuthorizer) Authorize(requestAttributes authorizer.Attributes) (bool, string, error) {
	rules, ruleResolutionError := r.authorizationRuleResolver.RulesFor(requestAttributes.GetUser(), requestAttributes.GetNamespace())
	if RulesAllow(requestAttributes, rules...) {
		return true, "", nil
	}

	// Build a detailed log of the denial.
	// Make the whole block conditional so we don't do a lot of string-building we won't use.
	if glog.V(2) {
		var operation string
		if requestAttributes.IsResourceRequest() {
			operation = fmt.Sprintf(
				"%q on \"%v.%v/%v\"",
				requestAttributes.GetVerb(),
				requestAttributes.GetResource(), requestAttributes.GetAPIGroup(), requestAttributes.GetSubresource(),
			)
		} else {
			operation = fmt.Sprintf("%q nonResourceURL %q", requestAttributes.GetVerb(), requestAttributes.GetPath())
		}

		var scope string
		if ns := requestAttributes.GetNamespace(); len(ns) > 0 {
			scope = fmt.Sprintf("in namespace %q", ns)
		} else {
			scope = "cluster-wide"
		}

		glog.Infof("RBAC DENY: user %q groups %v cannot %s %s", requestAttributes.GetUser().GetName(), requestAttributes.GetUser().GetGroups(), operation, scope)
	}

	reason := ""
	if ruleResolutionError != nil {
		reason = fmt.Sprintf("%v", ruleResolutionError)
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
	for _, rule := range rules {
		if RuleAllows(requestAttributes, rule) {
			return true
		}
	}

	return false
}

func RuleAllows(requestAttributes authorizer.Attributes, rule rbac.PolicyRule) bool {
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
