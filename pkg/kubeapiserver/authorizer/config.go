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

package authorizer

import (
	"errors"
	"fmt"
	"time"

	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/apiserver/pkg/authorization/authorizerfactory"
	"k8s.io/apiserver/pkg/authorization/union"
	"k8s.io/apiserver/plugin/pkg/authorizer/webhook"
	rbacapi "k8s.io/kubernetes/pkg/apis/rbac"
	"k8s.io/kubernetes/pkg/auth/authorizer/abac"
	"k8s.io/kubernetes/pkg/auth/nodeidentifier"
	informers "k8s.io/kubernetes/pkg/client/informers/informers_generated/internalversion"
	rbaclisters "k8s.io/kubernetes/pkg/client/listers/rbac/internalversion"
	"k8s.io/kubernetes/pkg/kubeapiserver/authorizer/modes"
	"k8s.io/kubernetes/plugin/pkg/auth/authorizer/node"
	"k8s.io/kubernetes/plugin/pkg/auth/authorizer/rbac"
	"k8s.io/kubernetes/plugin/pkg/auth/authorizer/rbac/bootstrappolicy"
)

type AuthorizationConfig struct {
	AuthorizationModes []string

	// Options for ModeABAC

	// Path to an ABAC policy file.
	PolicyFile string

	// Options for ModeWebhook

	// Kubeconfig file for Webhook authorization plugin.
	WebhookConfigFile string
	// TTL for caching of authorized responses from the webhook server.
	WebhookCacheAuthorizedTTL time.Duration
	// TTL for caching of unauthorized responses from the webhook server.
	WebhookCacheUnauthorizedTTL time.Duration

	InformerFactory informers.SharedInformerFactory
}

type roleGetter struct {
	lister rbaclisters.RoleLister
}

func (g *roleGetter) GetRole(namespace, name string) (*rbacapi.Role, error) {
	return g.lister.Roles(namespace).Get(name)
}

type roleBindingLister struct {
	lister rbaclisters.RoleBindingLister
}

func (l *roleBindingLister) ListRoleBindings(namespace string) ([]*rbacapi.RoleBinding, error) {
	return l.lister.RoleBindings(namespace).List(labels.Everything())
}

type clusterRoleGetter struct {
	lister rbaclisters.ClusterRoleLister
}

func (g *clusterRoleGetter) GetClusterRole(name string) (*rbacapi.ClusterRole, error) {
	return g.lister.Get(name)
}

type clusterRoleBindingLister struct {
	lister rbaclisters.ClusterRoleBindingLister
}

func (l *clusterRoleBindingLister) ListClusterRoleBindings() ([]*rbacapi.ClusterRoleBinding, error) {
	return l.lister.List(labels.Everything())
}

// New returns the right sort of union of multiple authorizer.Authorizer objects
// based on the authorizationMode or an error.
func (config AuthorizationConfig) New() (authorizer.Authorizer, error) {
	if len(config.AuthorizationModes) == 0 {
		return nil, errors.New("At least one authorization mode should be passed")
	}

	var authorizers []authorizer.Authorizer
	authorizerMap := make(map[string]bool)

	for _, authorizationMode := range config.AuthorizationModes {
		if authorizerMap[authorizationMode] {
			return nil, fmt.Errorf("Authorization mode %s specified more than once", authorizationMode)
		}
		// Keep cases in sync with constant list above.
		switch authorizationMode {
		case modes.ModeNode:
			graph := node.NewGraph()
			node.AddGraphEventHandlers(
				graph,
				config.InformerFactory.Core().InternalVersion().Pods(),
				config.InformerFactory.Core().InternalVersion().PersistentVolumes(),
			)
			nodeAuthorizer := node.NewAuthorizer(graph, nodeidentifier.NewDefaultNodeIdentifier(), bootstrappolicy.NodeRules())
			authorizers = append(authorizers, nodeAuthorizer)

			// Don't bind system:nodes to the system:node role
			bootstrappolicy.AddClusterRoleBindingFilter(bootstrappolicy.OmitNodesGroupBinding)

		case modes.ModeAlwaysAllow:
			authorizers = append(authorizers, authorizerfactory.NewAlwaysAllowAuthorizer())
		case modes.ModeAlwaysDeny:
			authorizers = append(authorizers, authorizerfactory.NewAlwaysDenyAuthorizer())
		case modes.ModeABAC:
			if config.PolicyFile == "" {
				return nil, errors.New("ABAC's authorization policy file not passed")
			}
			abacAuthorizer, err := abac.NewFromFile(config.PolicyFile)
			if err != nil {
				return nil, err
			}
			authorizers = append(authorizers, abacAuthorizer)
		case modes.ModeWebhook:
			if config.WebhookConfigFile == "" {
				return nil, errors.New("Webhook's configuration file not passed")
			}
			webhookAuthorizer, err := webhook.New(config.WebhookConfigFile,
				config.WebhookCacheAuthorizedTTL,
				config.WebhookCacheUnauthorizedTTL)
			if err != nil {
				return nil, err
			}
			authorizers = append(authorizers, webhookAuthorizer)
		case modes.ModeRBAC:
			rbacAuthorizer := rbac.New(
				&roleGetter{config.InformerFactory.Rbac().InternalVersion().Roles().Lister()},
				&roleBindingLister{config.InformerFactory.Rbac().InternalVersion().RoleBindings().Lister()},
				&clusterRoleGetter{config.InformerFactory.Rbac().InternalVersion().ClusterRoles().Lister()},
				&clusterRoleBindingLister{config.InformerFactory.Rbac().InternalVersion().ClusterRoleBindings().Lister()},
			)
			authorizers = append(authorizers, rbacAuthorizer)
		default:
			return nil, fmt.Errorf("Unknown authorization mode %s specified", authorizationMode)
		}
		authorizerMap[authorizationMode] = true
	}

	if !authorizerMap[modes.ModeABAC] && config.PolicyFile != "" {
		return nil, errors.New("Cannot specify --authorization-policy-file without mode ABAC")
	}
	if !authorizerMap[modes.ModeWebhook] && config.WebhookConfigFile != "" {
		return nil, errors.New("Cannot specify --authorization-webhook-config-file without mode Webhook")
	}

	return union.New(authorizers...), nil
}
