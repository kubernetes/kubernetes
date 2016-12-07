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

package options

import (
	"strings"
	"time"

	"github.com/spf13/pflag"

	authorizationclient "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/typed/authorization/v1beta1"
	"k8s.io/kubernetes/pkg/client/unversioned/clientcmd"
	"k8s.io/kubernetes/pkg/controller/informers"
	"k8s.io/kubernetes/pkg/genericapiserver/authorizer"
)

var AuthorizationModeChoices = []string{authorizer.ModeAlwaysAllow, authorizer.ModeAlwaysDeny, authorizer.ModeABAC, authorizer.ModeWebhook, authorizer.ModeRBAC}

type BuiltInAuthorizationOptions struct {
	Mode                        string
	PolicyFile                  string
	WebhookConfigFile           string
	WebhookCacheAuthorizedTTL   time.Duration
	WebhookCacheUnauthorizedTTL time.Duration
}

func NewBuiltInAuthorizationOptions() *BuiltInAuthorizationOptions {
	return &BuiltInAuthorizationOptions{
		Mode: authorizer.ModeAlwaysAllow,
		WebhookCacheAuthorizedTTL:   5 * time.Minute,
		WebhookCacheUnauthorizedTTL: 30 * time.Second,
	}
}

func (s *BuiltInAuthorizationOptions) Validate() []error {
	allErrors := []error{}
	return allErrors
}

func (s *BuiltInAuthorizationOptions) AddFlags(fs *pflag.FlagSet) {
	fs.StringVar(&s.Mode, "authorization-mode", s.Mode, ""+
		"Ordered list of plug-ins to do authorization on secure port. Comma-delimited list of: "+
		strings.Join(AuthorizationModeChoices, ",")+".")

	fs.StringVar(&s.PolicyFile, "authorization-policy-file", s.PolicyFile, ""+
		"File with authorization policy in csv format, used with --authorization-mode=ABAC, on the secure port.")

	fs.StringVar(&s.WebhookConfigFile, "authorization-webhook-config-file", s.WebhookConfigFile, ""+
		"File with webhook configuration in kubeconfig format, used with --authorization-mode=Webhook. "+
		"The API server will query the remote service to determine access on the API server's secure port.")

	fs.DurationVar(&s.WebhookCacheAuthorizedTTL, "authorization-webhook-cache-authorized-ttl",
		s.WebhookCacheAuthorizedTTL,
		"The duration to cache 'authorized' responses from the webhook authorizer. Default is 5m.")

	fs.DurationVar(&s.WebhookCacheUnauthorizedTTL,
		"authorization-webhook-cache-unauthorized-ttl", s.WebhookCacheUnauthorizedTTL,
		"The duration to cache 'unauthorized' responses from the webhook authorizer. Default is 30s.")

	fs.String("authorization-rbac-super-user", "", ""+
		"If specified, a username which avoids RBAC authorization checks and role binding "+
		"privilege escalation checks, to be used with --authorization-mode=RBAC.")
	fs.MarkDeprecated("authorization-rbac-super-user", "Removed during alpha to beta.  The 'system:masters' group has privileged access.")

}

func (s *BuiltInAuthorizationOptions) ToAuthorizationConfig(informerFactory informers.SharedInformerFactory) authorizer.AuthorizationConfig {
	modes := []string{}
	if len(s.Mode) > 0 {
		modes = strings.Split(s.Mode, ",")
	}

	return authorizer.AuthorizationConfig{
		AuthorizationModes:          modes,
		PolicyFile:                  s.PolicyFile,
		WebhookConfigFile:           s.WebhookConfigFile,
		WebhookCacheAuthorizedTTL:   s.WebhookCacheAuthorizedTTL,
		WebhookCacheUnauthorizedTTL: s.WebhookCacheUnauthorizedTTL,
		InformerFactory:             informerFactory,
	}
}

// DelegatingAuthorizationOptions provides an easy way for composing API servers to delegate their authorization to
// the root kube API server
type DelegatingAuthorizationOptions struct {
	// RemoteKubeConfigFile is the file to use to connect to a "normal" kube API server which hosts the
	// SubjectAccessReview.authorization.k8s.io endpoint for checking tokens.
	RemoteKubeConfigFile string

	// AllowCacheTTL is the length of time that a successful authorization response will be cached
	AllowCacheTTL time.Duration

	// DenyCacheTTL is the length of time that an unsuccessful authorization response will be cached.
	// You generally want more responsive, "deny, try again" flows.
	DenyCacheTTL time.Duration
}

func NewDelegatingAuthorizationOptions() *DelegatingAuthorizationOptions {
	return &DelegatingAuthorizationOptions{
		// very low for responsiveness, but high enough to handle storms
		AllowCacheTTL: 10 * time.Second,
		DenyCacheTTL:  10 * time.Second,
	}
}

func (s *DelegatingAuthorizationOptions) Validate() []error {
	allErrors := []error{}
	return allErrors
}

func (s *DelegatingAuthorizationOptions) AddFlags(fs *pflag.FlagSet) {
	fs.StringVar(&s.RemoteKubeConfigFile, "authorization-kubeconfig", s.RemoteKubeConfigFile, ""+
		"kubeconfig file pointing at the 'core' kubernetes server with enough rights to create "+
		" subjectaccessreviews.authorization.k8s.io.")

	fs.DurationVar(&s.AllowCacheTTL, "authorization-webhook-cache-authorized-ttl",
		s.AllowCacheTTL,
		"The duration to cache 'authorized' responses from the webhook authorizer.")

	fs.DurationVar(&s.DenyCacheTTL,
		"authorization-webhook-cache-unauthorized-ttl", s.DenyCacheTTL,
		"The duration to cache 'unauthorized' responses from the webhook authorizer.")
}

func (s *DelegatingAuthorizationOptions) ToAuthorizationConfig() (authorizer.DelegatingAuthorizerConfig, error) {
	sarClient, err := s.newSubjectAccessReview()
	if err != nil {
		return authorizer.DelegatingAuthorizerConfig{}, err
	}

	ret := authorizer.DelegatingAuthorizerConfig{
		SubjectAccessReviewClient: sarClient,
		AllowCacheTTL:             s.AllowCacheTTL,
		DenyCacheTTL:              s.DenyCacheTTL,
	}
	return ret, nil
}

func (s *DelegatingAuthorizationOptions) newSubjectAccessReview() (authorizationclient.SubjectAccessReviewInterface, error) {
	if len(s.RemoteKubeConfigFile) == 0 {
		return nil, nil
	}

	loadingRules := &clientcmd.ClientConfigLoadingRules{ExplicitPath: s.RemoteKubeConfigFile}
	loader := clientcmd.NewNonInteractiveDeferredLoadingClientConfig(loadingRules, &clientcmd.ConfigOverrides{})

	clientConfig, err := loader.ClientConfig()
	if err != nil {
		return nil, err
	}
	// set high qps/burst limits since this will effectively limit API server responsiveness
	clientConfig.QPS = 200
	clientConfig.Burst = 400

	client, err := authorizationclient.NewForConfig(clientConfig)
	if err != nil {
		return nil, err
	}

	return client.SubjectAccessReviews(), nil
}
