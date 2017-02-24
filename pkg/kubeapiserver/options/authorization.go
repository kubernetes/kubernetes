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

	informers "k8s.io/kubernetes/pkg/client/informers/informers_generated/internalversion"
	"k8s.io/kubernetes/pkg/kubeapiserver/authorizer"
	authzmodes "k8s.io/kubernetes/pkg/kubeapiserver/authorizer/modes"
)

type BuiltInAuthorizationOptions struct {
	Mode                        string
	PolicyFile                  string
	WebhookConfigFile           string
	WebhookCacheAuthorizedTTL   time.Duration
	WebhookCacheUnauthorizedTTL time.Duration
}

func NewBuiltInAuthorizationOptions() *BuiltInAuthorizationOptions {
	return &BuiltInAuthorizationOptions{
		Mode: authzmodes.ModeAlwaysAllow,
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
		strings.Join(authzmodes.AuthorizationModeChoices, ",")+".")

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
