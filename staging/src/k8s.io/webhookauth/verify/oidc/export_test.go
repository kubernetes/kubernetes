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

package oidc

import (
	"context"

	"k8s.io/webhookauth/verify"
)

// SetInClusterPathsForTest redirects the projected service-account token and CA
// bundle paths that InCluster reads. It exists only for tests: InCluster is
// intentionally option-free in production, but tests must point it at temporary
// files instead of the real /var/run/secrets projection. It returns a function
// that restores the previous paths.
func SetInClusterPathsForTest(tokenPath, caCertPath string) (restore func()) {
	prevToken, prevCA := serviceAccountTokenPath, serviceAccountCACertPath
	serviceAccountTokenPath = tokenPath
	serviceAccountCACertPath = caCertPath
	return func() {
		serviceAccountTokenPath = prevToken
		serviceAccountCACertPath = prevCA
	}
}

// NewDeferredVerifierForTest exposes the unexported deferred constructor (key set
// fetched now, audience bound later via verify.Verifier.BindAudience) so tests can
// exercise the in-cluster verifier shape before InCluster is rewired onto it.
func NewDeferredVerifierForTest(ctx context.Context, issuer string, opts ...Option) (*verify.Verifier, error) {
	return newDeferredVerifier(ctx, issuer, opts...)
}
