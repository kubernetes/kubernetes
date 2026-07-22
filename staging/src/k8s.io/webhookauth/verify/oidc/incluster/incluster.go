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

// Package incluster wires the OIDC token verifier for a webhook running inside a
// Kubernetes cluster, using the standard in-cluster REST config.
//
// It is deliberately the only package in this module that imports
// k8s.io/client-go, so out-of-cluster consumers of verify/oidc never pull in the
// client-go dependency tree.
package incluster // import "k8s.io/webhookauth/verify/oidc/incluster"

import (
	"context"
	"fmt"

	"k8s.io/client-go/rest"
	"k8s.io/webhookauth/verify"
	"k8s.io/webhookauth/verify/oidc"
)

// InCluster builds a [verify.Verifier] for a webhook running inside the cluster,
// with no static configuration. It reads the issuer and signing keys from the
// apiserver over the in-cluster network and defers audience binding, exactly as
// [oidc.NewLocalKeySetVerifier] documents (pair with
// admissionhttp.InClusterAudienceResolver). ctx governs the discovery fetch and
// background key refreshes, so pass a process-lifetime context.
func InCluster(ctx context.Context) (*verify.Verifier, error) {
	cfg, err := rest.InClusterConfig()
	if err != nil {
		return nil, fmt.Errorf("incluster: loading in-cluster REST config: %w", err)
	}
	httpClient, err := rest.HTTPClientFor(cfg)
	if err != nil {
		return nil, fmt.Errorf("incluster: building in-cluster HTTP client: %w", err)
	}
	// Reach the apiserver at the canonical in-cluster FQDN rather than cfg.Host:
	// the FQDN resolves on both Linux and Windows nodes, and the client-go
	// transport already trusts the cluster CA for that name.
	return oidc.NewLocalKeySetVerifier(ctx, oidc.InClusterAPIServerURL, oidc.WithHTTPClient(httpClient))
}
