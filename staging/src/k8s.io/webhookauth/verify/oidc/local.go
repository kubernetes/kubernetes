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
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"

	coreosoidc "github.com/coreos/go-oidc"
	"k8s.io/webhookauth/verify"
)

const (
	// InClusterAPIServerURL is the in-cluster address of the Kubernetes apiserver.
	//
	// The fully qualified ".cluster.local" form is REQUIRED: on Windows nodes the
	// short "kubernetes.default.svc" name does not resolve reliably, whereas the
	// FQDN does on both Linux and Windows. It is exported so the in-cluster
	// callers (the incluster package and admissionhttp's in-cluster handler) pass
	// a single canonical value instead of a per-caller rest.Config host.
	InClusterAPIServerURL = "https://kubernetes.default.svc.cluster.local"

	// openIDConfigPath is the apiserver's OIDC discovery endpoint.
	openIDConfigPath = "/.well-known/openid-configuration"
	// localJWKSPath is the apiserver's local JWKS endpoint. Keys are fetched here
	// rather than via the discovery document's jwks_uri, which may point at an
	// external address the pod cannot reach.
	localJWKSPath = "/openid/v1/jwks"

	// maxDiscoveryBytes bounds the discovery-document read.
	maxDiscoveryBytes = 1 << 20 // 1 MiB

	// serviceAccountCAPath is the projected service account CA bundle every pod
	// receives; it is the CA that signs the in-cluster apiserver's serving cert.
	serviceAccountCAPath = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"
)

// NewLocalKeySetVerifier builds a deferred [verify.Verifier] for a webhook that
// reaches its apiserver at apiServerURL. It reads the issuer from the apiserver's
// OIDC discovery document and fetches signing keys from its local JWKS endpoint,
// so the request never leaves the cluster network by following an external
// issuer URL.
//
// The audience is derived at runtime from the first admission request (see
// admissionhttp.InClusterAudienceResolver), so until one is bound the verifier
// denies every token and reports unhealthy. ctx governs the discovery fetch and
// background key refreshes, so pass a process-lifetime context. When no
// [WithHTTPClient] is supplied a client trusting the projected service account CA
// bundle is built automatically, so an in-cluster caller can omit the transport.
func NewLocalKeySetVerifier(ctx context.Context, apiServerURL string, opts ...Option) (*verify.Verifier, error) {
	if apiServerURL == "" {
		return nil, errors.New("oidc: apiServerURL must not be empty")
	}
	apiServerURL = strings.TrimRight(apiServerURL, "/")

	cfg := &config{}
	for _, opt := range opts {
		opt(cfg)
	}
	httpClient := cfg.httpClient
	if httpClient == nil {
		// No transport was supplied: build one that trusts the cluster CA so the
		// in-cluster apiserver's serving certificate validates. Fail closed if the
		// CA bundle cannot be loaded rather than fall back to an untrusted client.
		var err error
		httpClient, err = inClusterHTTPClient()
		if err != nil {
			return nil, err
		}
	}

	issuer, err := fetchLocalIssuer(ctx, httpClient, apiServerURL+openIDConfigPath)
	if err != nil {
		return nil, err
	}

	// go-oidc reads the HTTP client from the context for background key refreshes;
	// use the same resolved client that fetched discovery so key refreshes trust
	// the cluster CA too.
	keyCtx := coreosoidc.ClientContext(ctx, httpClient)
	// Keys come from the local endpoint, not the discovery jwks_uri, so the fetch
	// stays in-cluster.
	keySet := coreosoidc.NewRemoteKeySet(keyCtx, apiServerURL+localJWKSPath)

	return verify.NewVerifier(&oidcAuthenticator{issuer: issuer, keySet: keySet})
}

// inClusterHTTPClient builds an HTTP client whose TLS config trusts only the
// projected service account CA bundle, which signs the in-cluster apiserver's
// serving certificate. It fails closed if the bundle is missing or unparseable,
// so a misconfigured pod cannot silently fall back to an untrusted transport.
//
// This is production-only wiring: the CA bundle path exists only inside a running
// pod, so it cannot be exercised by the module's tests, which always inject a
// transport via WithHTTPClient.
func inClusterHTTPClient() (*http.Client, error) {
	caPEM, err := os.ReadFile(serviceAccountCAPath)
	if err != nil {
		return nil, fmt.Errorf("oidc: reading in-cluster CA bundle %q: %w", serviceAccountCAPath, err)
	}
	pool := x509.NewCertPool()
	if !pool.AppendCertsFromPEM(caPEM) {
		return nil, fmt.Errorf("oidc: in-cluster CA bundle %q contained no valid certificates", serviceAccountCAPath)
	}
	return &http.Client{
		Transport: &http.Transport{
			TLSClientConfig: &tls.Config{
				RootCAs:    pool,
				MinVersion: tls.VersionTLS12,
			},
		},
	}, nil
}

// fetchLocalIssuer GETs the OIDC discovery document at discoveryURL and returns
// its "issuer". A non-200 response, an unreadable or malformed body, or an empty
// issuer is an error.
func fetchLocalIssuer(ctx context.Context, client *http.Client, discoveryURL string) (string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, discoveryURL, nil)
	if err != nil {
		return "", fmt.Errorf("oidc: building discovery request: %w", err)
	}
	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("oidc: fetching discovery document: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("oidc: discovery document request returned %s", resp.Status)
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, maxDiscoveryBytes))
	if err != nil {
		return "", fmt.Errorf("oidc: reading discovery document: %w", err)
	}
	var doc struct {
		Issuer string `json:"issuer"`
	}
	if err := json.Unmarshal(body, &doc); err != nil {
		return "", fmt.Errorf("oidc: parsing discovery document: %w", err)
	}
	if doc.Issuer == "" {
		return "", errors.New("oidc: discovery document has no issuer")
	}
	return doc.Issuer, nil
}
