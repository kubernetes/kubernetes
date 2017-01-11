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

	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/apiserver/pkg/authorization/union"
	"k8s.io/kubernetes/pkg/auth/authorizer/abac"
	"k8s.io/kubernetes/pkg/controller/informers"
	genericauthorizer "k8s.io/kubernetes/pkg/genericapiserver/authorizer"
	"k8s.io/kubernetes/plugin/pkg/auth/authorizer/rbac"
	"k8s.io/kubernetes/plugin/pkg/auth/authorizer/webhook"
)

const (
	ModeAlwaysAllow string = "AlwaysAllow"
	ModeAlwaysDeny  string = "AlwaysDeny"
	ModeABAC        string = "ABAC"
	ModeWebhook     string = "Webhook"
	ModeRBAC        string = "RBAC"
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

	// Options for RBAC

	// User which can bootstrap role policies
	RBACSuperUser string

	InformerFactory informers.SharedInformerFactory
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
		case ModeAlwaysAllow:
			authorizers = append(authorizers, genericauthorizer.NewAlwaysAllowAuthorizer())
		case ModeAlwaysDeny:
			authorizers = append(authorizers, genericauthorizer.NewAlwaysDenyAuthorizer())
		case ModeABAC:
			if config.PolicyFile == "" {
				return nil, errors.New("ABAC's authorization policy file not passed")
			}
			abacAuthorizer, err := abac.NewFromFile(config.PolicyFile)
			if err != nil {
				return nil, err
			}
			authorizers = append(authorizers, abacAuthorizer)
		case ModeWebhook:
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
		case ModeRBAC:
			rbacAuthorizer := rbac.New(
				config.InformerFactory.Roles().Lister(),
				config.InformerFactory.RoleBindings().Lister(),
				config.InformerFactory.ClusterRoles().Lister(),
				config.InformerFactory.ClusterRoleBindings().Lister(),
			)
			authorizers = append(authorizers, rbacAuthorizer)
		default:
			return nil, fmt.Errorf("Unknown authorization mode %s specified", authorizationMode)
		}
		authorizerMap[authorizationMode] = true
	}

	if !authorizerMap[ModeABAC] && config.PolicyFile != "" {
		return nil, errors.New("Cannot specify --authorization-policy-file without mode ABAC")
	}
	if !authorizerMap[ModeWebhook] && config.WebhookConfigFile != "" {
		return nil, errors.New("Cannot specify --authorization-webhook-config-file without mode Webhook")
	}
	if !authorizerMap[ModeRBAC] && config.RBACSuperUser != "" {
		return nil, errors.New("Cannot specify --authorization-rbac-super-user without mode RBAC")
	}

	return union.New(authorizers...), nil
}
