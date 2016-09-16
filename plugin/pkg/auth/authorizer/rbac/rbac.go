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
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/rbac"
	"k8s.io/kubernetes/pkg/apis/rbac/validation"
	"k8s.io/kubernetes/pkg/auth/authorizer"
	"k8s.io/kubernetes/pkg/registry/clusterrole"
	"k8s.io/kubernetes/pkg/registry/clusterrolebinding"
	"k8s.io/kubernetes/pkg/registry/role"
	"k8s.io/kubernetes/pkg/registry/rolebinding"
)

type RBACAuthorizer struct {
	superUser string

	authorizationRuleResolver validation.AuthorizationRuleResolver
}

func (r *RBACAuthorizer) Authorize(attr authorizer.Attributes) (bool, string, error) {
	if r.superUser != "" && attr.GetUser() != nil && attr.GetUser().GetName() == r.superUser {
		return true, "", nil
	}

	ctx := api.WithNamespace(api.WithUser(api.NewContext(), attr.GetUser()), attr.GetNamespace())

	// Frame the authorization request as a privilege escalation check.
	var requestedRule rbac.PolicyRule
	if attr.IsResourceRequest() {
		resource := attr.GetResource()
		if len(attr.GetSubresource()) > 0 {
			resource = attr.GetResource() + "/" + attr.GetSubresource()
		}
		requestedRule = rbac.PolicyRule{
			Verbs:         []string{attr.GetVerb()},
			APIGroups:     []string{attr.GetAPIGroup()}, // TODO(ericchiang): add api version here too?
			Resources:     []string{resource},
			ResourceNames: []string{attr.GetName()},
		}
	} else {
		requestedRule = rbac.PolicyRule{
			Verbs:           []string{attr.GetVerb()},
			NonResourceURLs: []string{attr.GetPath()},
		}
	}

	// TODO(nhlfr): Try to find more lightweight way to check attributes than escalation checks.
	err := validation.ConfirmNoEscalation(ctx, r.authorizationRuleResolver, []rbac.PolicyRule{requestedRule})
	if err != nil {
		return false, err.Error(), nil
	}

	return true, "", nil
}

func New(roleRegistry role.Registry, roleBindingRegistry rolebinding.Registry, clusterRoleRegistry clusterrole.Registry, clusterRoleBindingRegistry clusterrolebinding.Registry, superUser string) *RBACAuthorizer {
	authorizer := &RBACAuthorizer{
		superUser: superUser,
		authorizationRuleResolver: validation.NewDefaultRuleResolver(
			roleRegistry,
			roleBindingRegistry,
			clusterRoleRegistry,
			clusterRoleBindingRegistry,
		),
	}
	return authorizer
}
