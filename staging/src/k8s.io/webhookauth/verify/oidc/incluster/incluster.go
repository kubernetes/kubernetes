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
// It is deliberately the ONLY package in this module that imports
// k8s.io/client-go. Keeping the client-go dependency confined here means
// out-of-cluster consumers of verify/oidc (who call NewRemoteVerifier or supply
// their own transport) never pull in the client-go dependency tree. The
// module-root .import-restrictions forbids client-go everywhere; a local
// .import-restrictions in this directory re-permits it for this package only.
package incluster // import "k8s.io/webhookauth/verify/oidc/incluster"

import (
	"context"
	"fmt"

	"k8s.io/client-go/rest"
	"k8s.io/webhookauth/verify"
	"k8s.io/webhookauth/verify/oidc"
)

// InCluster builds a [verify.Verifier] for a webhook running inside the cluster,
// with no static configuration. It uses the standard in-cluster REST config for
// the apiserver address and a cluster-CA-trusting transport, then reads the token
// issuer from the apiserver's OIDC discovery document and fetches signing keys
// from the apiserver's local JWKS endpoint — all over the in-cluster network (see
// [oidc.NewLocalKeySetVerifier]).
//
// The expected audience is not known here: an in-cluster webhook derives it from
// the first admission request (pair this with
// admissionhttp.InClusterAudienceResolver). The returned verifier therefore
// denies every token and reports unhealthy via [verify.Verifier.HealthCheck]
// until an audience is bound — the signal a controller-runtime health check turns
// into a restart if the audience can never be derived.
//
// ctx governs the discovery fetch and the key set's background refreshes, so pass
// the process-lifetime context.
func InCluster(ctx context.Context) (*verify.Verifier, error) {
	cfg, err := rest.InClusterConfig()
	if err != nil {
		return nil, fmt.Errorf("incluster: loading in-cluster REST config: %w", err)
	}
	httpClient, err := rest.HTTPClientFor(cfg)
	if err != nil {
		return nil, fmt.Errorf("incluster: building in-cluster HTTP client: %w", err)
	}
	return oidc.NewLocalKeySetVerifier(ctx, cfg.Host, oidc.WithHTTPClient(httpClient))
}
