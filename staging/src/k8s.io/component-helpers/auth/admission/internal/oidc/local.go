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
	"net/url"
	"strings"
	"time"

	"k8s.io/client-go/rest"
	"k8s.io/component-helpers/auth/admission/internal/verify"
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

	// serviceAccountCAPath is the projected service account CA bundle every pod
	// receives; it is the CA that signs the in-cluster apiserver's serving cert.
	serviceAccountCAPath = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"
)

// NewLocalKeySetVerifier discovers the issuer and signing keys from the local
// apiserver, using the expected audience supplied from the webhook configuration.
// A missing audience is an error even when discovery and TLS are in-cluster.
// A removed key may remain usable for at most 60 seconds while its snapshot is
// fresh; an unavailable apiserver after that deadline fails closed.
func NewLocalKeySetVerifier(ctx context.Context, apiServerURL, audience string, httpClient *http.Client) (*verify.Verifier, error) {
	authenticator, err := newLocalAuthenticator(ctx, apiServerURL, audience, httpClient, time.Now)
	if err != nil {
		return nil, err
	}
	v, err := verify.NewVerifier(authenticator)
	if err != nil {
		return nil, err
	}
	if err := v.BindAudience(audience); err != nil {
		return nil, err
	}
	return v, nil
}

func newLocalAuthenticator(ctx context.Context, apiServerURL, audience string, httpClient *http.Client, now func() time.Time) (*oidcAuthenticator, error) {
	if audience == "" {
		return nil, errors.New("oidc: the webhook's configured audience must not be empty")
	}
	if httpClient == nil {
		var err error
		var inClusterURL string
		inClusterURL, httpClient, err = inClusterHTTPClient()
		if err != nil {
			return nil, err
		}
		if apiServerURL == "" {
			apiServerURL = inClusterURL
		} else if strings.TrimRight(apiServerURL, "/") != strings.TrimRight(inClusterURL, "/") {
			return nil, errors.New("oidc: apiServerURL must match the in-cluster apiserver when using projected ServiceAccount credentials")
		}
	}
	if apiServerURL == "" {
		return nil, errors.New("oidc: apiServerURL must not be empty when an HTTP client is supplied")
	}
	apiServerURL = strings.TrimRight(apiServerURL, "/")
	server, err := url.Parse(apiServerURL)
	if err != nil || server.Scheme != "https" || server.Host == "" || server.User != nil || server.Path != "" || server.RawQuery != "" || server.Fragment != "" {
		return nil, fmt.Errorf("oidc: apiServerURL %q must be an HTTPS origin", apiServerURL)
	}

	// The issuer may advertise an external JWKS URI, but only the fixed local
	// apiserver endpoint and redirects within that same TLS origin are trusted.
	boundedClient := boundedHTTPClient(httpClient)
	previousRedirectPolicy := boundedClient.CheckRedirect
	boundedClient.CheckRedirect = func(req *http.Request, via []*http.Request) error {
		if req.URL.Scheme != server.Scheme || req.URL.Host != server.Host {
			return errors.New("oidc: local apiserver redirect leaves the trusted origin")
		}
		return previousRedirectPolicy(req, via)
	}

	ctx, cancel := context.WithTimeout(ctx, signingKeyRefreshTimeout)
	defer cancel()
	refreshStartedAt := now()
	issuer, err := fetchLocalIssuer(ctx, boundedClient, apiServerURL+openIDConfigPath)
	if err != nil {
		return nil, err
	}
	readiness, err := newKeySetReadiness(apiServerURL+localJWKSPath, boundedClient, serviceAccountSigningAlgs)
	if err != nil {
		return nil, err
	}
	keys, err := readiness.loadKeys(ctx)
	if err != nil {
		return nil, fmt.Errorf("oidc: local apiserver signing keys are not ready: %w", err)
	}
	loadKeys := func(ctx context.Context) (*jwksKeySet, []string, error) {
		refreshedIssuer, err := fetchLocalIssuer(ctx, boundedClient, apiServerURL+openIDConfigPath)
		if err != nil {
			return nil, nil, err
		}
		if refreshedIssuer != issuer {
			return nil, nil, fmt.Errorf("oidc: local apiserver issuer changed from %q to %q", issuer, refreshedIssuer)
		}
		keys, err := readiness.loadKeys(ctx)
		if err != nil {
			return nil, nil, err
		}
		return keys, serviceAccountSigningAlgs, nil
	}
	return &oidcAuthenticator{newRefreshingAuthenticator(issuer, audience, keys, serviceAccountSigningAlgs, loadKeys, now, refreshStartedAt)}, nil
}

// inClusterHTTPClient retains the projected ServiceAccount bearer because the
// apiserver grants OIDC discovery to service accounts, not anonymous callers.
// newLocalAuthenticator pins every request and redirect to this apiserver origin.
func inClusterHTTPClient() (string, *http.Client, error) {
	cfg, err := rest.InClusterConfig()
	if err != nil {
		return "", nil, fmt.Errorf("oidc: loading in-cluster ServiceAccount configuration: %w", err)
	}
	client, err := serviceAccountTLSClientForConfig(cfg)
	if err != nil {
		return "", nil, err
	}
	return cfg.Host, client, nil
}

func serviceAccountTLSClientForConfig(cfg *rest.Config) (*http.Client, error) {
	// InClusterConfig logs CA failures but can still return a config. Refuse
	// ambient system roots rather than quietly dropping the apiserver trust pin.
	if cfg == nil || cfg.TLSClientConfig.Insecure || cfg.TLSClientConfig.CAFile == "" && len(cfg.TLSClientConfig.CAData) == 0 {
		return nil, fmt.Errorf("oidc: projected ServiceAccount CA %q is missing or invalid", serviceAccountCAPath)
	}
	client, err := rest.HTTPClientFor(cfg)
	if err != nil {
		return nil, fmt.Errorf("oidc: creating trusted in-cluster HTTP client: %w", err)
	}
	return client, nil
}

// fetchLocalIssuer GETs the OIDC discovery document at discoveryURL and returns
// its "issuer". A non-200 response, an unreadable or malformed body, or an empty
// issuer is an error.
func fetchLocalIssuer(ctx context.Context, client *http.Client, discoveryURL string) (string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, discoveryURL, nil)
	if err != nil {
		return "", fmt.Errorf("oidc: building discovery request: %w", err)
	}
	req.Header.Set("Cache-Control", "no-cache")
	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("oidc: fetching discovery document: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("oidc: discovery document request returned %s", resp.Status)
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, maxDiscoveryBytes+1))
	if err != nil {
		return "", fmt.Errorf("oidc: reading discovery document: %w", err)
	}
	if len(body) > maxDiscoveryBytes {
		return "", errors.New("oidc: discovery document exceeds the size limit")
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
