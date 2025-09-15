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

package authorizerfactory

import (
	"errors"
	"time"

	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	authorizationcel "k8s.io/apiserver/pkg/authorization/cel"
	"k8s.io/apiserver/plugin/pkg/authorizer/webhook"
	authorizationclient "k8s.io/client-go/kubernetes/typed/authorization/v1"
)

// DelegatingAuthorizerConfig is the minimal configuration needed to create an authorizer
// built to delegate authorization to a kube API server
type DelegatingAuthorizerConfig struct {
	SubjectAccessReviewClient authorizationclient.AuthorizationV1Interface

	// Compiler is the CEL compiler to use for evaluating policies. If nil, a default compiler will be used.
	Compiler authorizationcel.Compiler

	// AllowCacheTTL is the length of time that a successful authorization response will be cached
	AllowCacheTTL time.Duration

	// DenyCacheTTL is the length of time that an unsuccessful authorization response will be cached.
	// You generally want more responsive, "deny, try again" flows.
	DenyCacheTTL time.Duration

	// WebhookRetryBackoff specifies the backoff parameters for the authorization webhook retry logic.
	// This allows us to configure the sleep time at each iteration and the maximum number of retries allowed
	// before we fail the webhook call in order to limit the fan out that ensues when the system is degraded.
	WebhookRetryBackoff *wait.Backoff
}

func (c DelegatingAuthorizerConfig) New() (authorizer.Authorizer, error) {
	if c.WebhookRetryBackoff == nil {
		return nil, errors.New("retry backoff parameters for delegating authorization webhook has not been specified")
	}
	compiler := c.Compiler
	if compiler == nil {
		compiler = authorizationcel.NewDefaultCompiler()
	}

	return webhook.NewFromInterface(
		c.SubjectAccessReviewClient,
		c.AllowCacheTTL,
		c.DenyCacheTTL,
		*c.WebhookRetryBackoff,
		authorizer.DecisionNoOpinion,
		NewDelegatingAuthorizerMetrics(),
		compiler,
	)
}
