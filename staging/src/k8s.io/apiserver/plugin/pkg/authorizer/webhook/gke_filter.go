/*
Copyright 2018 The Kubernetes Authors.

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

package webhook

import (
	"context"
	"fmt"
	"regexp"
	"time"

	"k8s.io/klog/v2"

	utilnet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/client-go/tools/clientcmd"
)

// Skip authorize calls to hostedmaster if the answer is known.

// Mitigate scalability problems related to Webhook Authorizer, that sends too many requests to hostedmaster.
// All requests that were not authorized by Node and RBAC Authorizers are passed to webhook.
// One scenario: when kube-apiserver is under heavy load (cluster creation or scale up),
// Node Authorize fails to keep up with new nodes causing kubelet authorization to be passed to webhook.
// That triggers hostedmaster DoS protection, further degrading responsiveness.

// authorizeUrlRegex determines authorize URLs for which to use GKE custom logic
// Protects possible ONYX deployments with different authorization logic
const authorizeUrlRegex = `^https://[^/]*container.(sandbox.)?googleapis.com/`

// Map key for storing userAssertions in the User.Extra field of the
// Authentication response.
const userAssertionKey = "user-assertion.cloud.google.com"

// hasUserAssertion should replicate logic of hostedmaster:
// http://google3/cloud/kubernetes/engine/hostedmaster/hostedmasterserver.go?l=1105&rcl=352896285
func hasUserAssertion(attr authorizer.Attributes) bool {
	user := attr.GetUser()
	if user == nil {
		return false
	}
	extra := user.GetExtra()
	if extra == nil {
		return false
	}
	_, exist := extra[userAssertionKey]
	return exist
}

type RuleResolverAuthorizer interface {
	authorizer.Authorizer
	authorizer.RuleResolver
}

// Ensure Webhook implements needed interfaces.
var _ RuleResolverAuthorizer = (*GkeApiserverWebhookAuthorizer)(nil)

type GkeApiserverWebhookAuthorizer struct {
	webhookAuthorizer RuleResolverAuthorizer
}

func NewGkeApiserverWebhookAuthorizer(kubeConfigFile string, version string, authorizedTTL, unauthorizedTTL time.Duration, retryBackoff wait.Backoff, customDial utilnet.DialFunc) (RuleResolverAuthorizer, error) {
	// Create regular WebhookAuthorizer
	webhookAuthorizer, err := New(kubeConfigFile, version, authorizedTTL, unauthorizedTTL, retryBackoff, customDial)
	if err != nil {
		return nil, err
	}
	// Check if running on GKE
	isGke, err := isUsingGkeHostedMaster(kubeConfigFile)
	if err != nil {
		klog.Errorf("Failed to extract authorize URL: %v", err)
	}
	if !isGke {
		klog.Info("Using generic Webhook Authorizer logic")
		return webhookAuthorizer, nil
	}
	// Decorate with custom GKE authorizer
	klog.Info("Enabling GKE-only custom logic in Webhook Authorizer")
	return &GkeApiserverWebhookAuthorizer{webhookAuthorizer}, nil
}

// isUsingGkeHostedMaster checks if authz config uses URL that matches GKE hostedmaster
func isUsingGkeHostedMaster(kubeConfigFile string) (bool, error) {
	loadingRules := clientcmd.NewDefaultClientConfigLoadingRules()
	loadingRules.ExplicitPath = kubeConfigFile
	loader := clientcmd.NewNonInteractiveDeferredLoadingClientConfig(loadingRules, &clientcmd.ConfigOverrides{})
	clientConfig, err := loader.ClientConfig()
	if err != nil {
		return false, fmt.Errorf("failed to load config: %v", err)
	}

	return isGkeHostedMasterUrl(clientConfig.Host)
}

// isGkeHostedMasterUrl checks if URL matches GKE hostedmaster
func isGkeHostedMasterUrl(authorizeUrl string) (bool, error) {
	matched, err := regexp.MatchString(authorizeUrlRegex, authorizeUrl)
	if err != nil {
		return false, err
	}
	return matched, nil
}

func (g *GkeApiserverWebhookAuthorizer) Authorize(ctx context.Context, attr authorizer.Attributes) (decision authorizer.Decision, reason string, err error) {
	// b/142480707 Filter apiserver kube-apiserver unanswerable calls to hostedmaster
	if !hasUserAssertion(attr) {
		return authorizer.DecisionNoOpinion, "", nil
	}
	// Fallback to OSS logic
	return g.webhookAuthorizer.Authorize(ctx, attr)
}

func (g *GkeApiserverWebhookAuthorizer) RulesFor(user user.Info, namespace string) ([]authorizer.ResourceRuleInfo, []authorizer.NonResourceRuleInfo, bool, error) {
	return g.webhookAuthorizer.RulesFor(user, namespace)
}
