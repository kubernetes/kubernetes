/*
Copyright The Kubernetes Authors.

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
	"net"

	"k8s.io/apimachinery/pkg/util/wait"
	servingoptions "k8s.io/apiserver/pkg/server/serving/options"
)

// The secure serving and delegated authentication and authorization options
// live in k8s.io/apiserver/pkg/server/serving/options so that components which
// only serve HTTPS do not link the generic API server. The declarations below
// keep existing importers of this package compiling.

type (
	SecureServingOptions               = servingoptions.SecureServingOptions
	SecureServingOptionsWithLoopback   = servingoptions.SecureServingOptionsWithLoopback
	CertKey                            = servingoptions.CertKey
	GeneratableKeyCert                 = servingoptions.GeneratableKeyCert
	HealthzLivezHealthChecksAdder      = servingoptions.HealthzLivezHealthChecksAdder
	LoopbackCertificateExpiredError    = servingoptions.LoopbackCertificateExpiredError
	RequestHeaderAuthenticationOptions = servingoptions.RequestHeaderAuthenticationOptions
	ClientCertAuthenticationOptions    = servingoptions.ClientCertAuthenticationOptions
	DelegatingAuthenticationOptions    = servingoptions.DelegatingAuthenticationOptions
	DynamicRequestHeaderController     = servingoptions.DynamicRequestHeaderController
	DelegatingAuthorizationOptions     = servingoptions.DelegatingAuthorizationOptions
)

// NewSecureServingOptions returns the default secure serving options.
func NewSecureServingOptions() *SecureServingOptions {
	return servingoptions.NewSecureServingOptions()
}

// NewDelegatingAuthenticationOptions returns the default delegating authentication options.
func NewDelegatingAuthenticationOptions() *DelegatingAuthenticationOptions {
	return servingoptions.NewDelegatingAuthenticationOptions()
}

// NewDelegatingAuthorizationOptions returns the default delegating authorization options.
func NewDelegatingAuthorizationOptions() *DelegatingAuthorizationOptions {
	return servingoptions.NewDelegatingAuthorizationOptions()
}

// DefaultAuthWebhookRetryBackoff is the default backoff parameters for
// both authentication and authorization webhook used by the apiserver.
func DefaultAuthWebhookRetryBackoff() *wait.Backoff {
	return servingoptions.DefaultAuthWebhookRetryBackoff()
}

// CreateListener creates a listener on addr, or on an ephemeral port when addr ends in :0.
func CreateListener(network, addr string, config net.ListenConfig) (net.Listener, int, error) {
	return servingoptions.CreateListener(network, addr, config)
}
