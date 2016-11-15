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
	"k8s.io/kubernetes/pkg/apis/rbac"
	"k8s.io/kubernetes/pkg/apis/rbac/validation"
	"k8s.io/kubernetes/pkg/auth/authorizer"
	"k8s.io/kubernetes/pkg/auth/user"
)

type RequestToRuleMapper interface {
	// RulesFor returns all known PolicyRules and any errors that happened while locating those rules.
	// Any rule returned is still valid, since rules are deny by default.  If you can pass with the rules
	// supplied, you do not have to fail the request.  If you cannot, you should indicate the error along
	// with your denial.
	RulesFor(subject user.Info, namespace string) ([]rbac.PolicyRule, error)
}

type RBACAuthorizer struct {
	superUser string

	authorizationRuleResolver RequestToRuleMapper
}

func (r *RBACAuthorizer) Authorize(requestAttributes authorizer.Attributes) (bool, string, error) {
	if r.superUser != "" && requestAttributes.GetUser() != nil && requestAttributes.GetUser().GetName() == r.superUser {
		return true, "", nil
	}

	rules, ruleResolutionError := r.authorizationRuleResolver.RulesFor(requestAttributes.GetUser(), requestAttributes.GetNamespace())
	if RulesAllow(requestAttributes, rules...) {
		return true, "", nil
	}

	return false, "", ruleResolutionError
}

func New(roles validation.RoleGetter, roleBindings validation.RoleBindingLister, clusterRoles validation.ClusterRoleGetter, clusterRoleBindings validation.ClusterRoleBindingLister, superUser string) *RBACAuthorizer {
	authorizer := &RBACAuthorizer{
		superUser: superUser,
		authorizationRuleResolver: validation.NewDefaultRuleResolver(
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
