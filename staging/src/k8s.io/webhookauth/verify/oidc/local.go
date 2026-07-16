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
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"

	coreosoidc "github.com/coreos/go-oidc"
	"k8s.io/webhookauth/verify"
)

const (
	// openIDConfigPath is the apiserver's OIDC discovery endpoint.
	openIDConfigPath = "/.well-known/openid-configuration"
	// localJWKSPath is the apiserver's local JWKS endpoint. Keys are fetched here
	// rather than via the discovery document's jwks_uri, which may point at an
	// external address the pod cannot reach.
	localJWKSPath = "/openid/v1/jwks"

	// maxDiscoveryBytes bounds the discovery-document read.
	maxDiscoveryBytes = 1 << 20 // 1 MiB
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
// background key refreshes, so pass a process-lifetime context, and supply
// [WithHTTPClient] with a transport that trusts the cluster CA.
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
		httpClient = http.DefaultClient
	}

	issuer, err := fetchLocalIssuer(ctx, httpClient, apiServerURL+openIDConfigPath)
	if err != nil {
		return nil, err
	}

	// go-oidc reads the HTTP client from the context for background key refreshes.
	keyCtx := ctx
	if cfg.httpClient != nil {
		keyCtx = coreosoidc.ClientContext(ctx, cfg.httpClient)
	}
	// Keys come from the local endpoint, not the discovery jwks_uri, so the fetch
	// stays in-cluster.
	keySet := coreosoidc.NewRemoteKeySet(keyCtx, apiServerURL+localJWKSPath)

	return verify.NewVerifier(&oidcAuthenticator{issuer: issuer, keySet: keySet})
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
