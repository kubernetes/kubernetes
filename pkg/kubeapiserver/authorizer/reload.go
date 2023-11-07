/*
Copyright 2024 The Kubernetes Authors.

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

package authorizer

import (
	"context"
	"sync/atomic"

	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/kubernetes/pkg/auth/authorizer/abac"
	"k8s.io/kubernetes/plugin/pkg/auth/authorizer/node"
	"k8s.io/kubernetes/plugin/pkg/auth/authorizer/rbac"
)

type reloadableAuthorizerResolver struct {
	// initialConfig holds the ReloadFile used to initiate background reloading,
	// and information used to construct webhooks that isn't exposed in the authorization
	// configuration file (dial function, backoff settings, etc)
	initialConfig Config

	nodeAuthorizer *node.NodeAuthorizer
	rbacAuthorizer *rbac.RBACAuthorizer
	abacAuthorizer abac.PolicyList

	current atomic.Pointer[authorizerResolver]
}

type authorizerResolver struct {
	authorizer   authorizer.Authorizer
	ruleResolver authorizer.RuleResolver
}

func (r *reloadableAuthorizerResolver) Authorize(ctx context.Context, a authorizer.Attributes) (authorized authorizer.Decision, reason string, err error) {
	return r.current.Load().authorizer.Authorize(ctx, a)
}

func (r *reloadableAuthorizerResolver) RulesFor(user user.Info, namespace string) ([]authorizer.ResourceRuleInfo, []authorizer.NonResourceRuleInfo, bool, error) {
	return r.current.Load().ruleResolver.RulesFor(user, namespace)
}
