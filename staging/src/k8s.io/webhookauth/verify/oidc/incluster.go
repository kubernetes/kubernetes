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
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"strings"

	"k8s.io/webhookauth/verify"
)

// serviceAccountTokenPath and serviceAccountCACertPath are where the kubelet
// projects the pod's service-account token and the cluster CA bundle. An
// in-cluster webhook always holds both. They are package variables (not
// constants) solely so tests can redirect them; production always uses the
// projected paths.
var (
	serviceAccountTokenPath  = "/var/run/secrets/kubernetes.io/serviceaccount/token"
	serviceAccountCACertPath = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"
)

// InCluster builds a [verify.Verifier] for a webhook running inside the cluster,
// with no configuration required. It:
//
//   - reads the issuer from the "iss" claim of the pod's own projected
//     service-account token,
//   - builds an HTTP client that trusts the cluster CA (from the projected
//     ca.crt) so OIDC discovery and JWKS retrieval succeed over the in-cluster
//     HTTPS endpoint, and
//   - derives a PROVISIONAL expected audience from the "aud" of that same
//     service-account token.
//
// This is intentionally option-free: the whole point of the in-cluster path is
// that the cluster already projects everything needed. Deployments that need to
// override the issuer, audience, or transport should call [NewRemoteVerifier]
// directly.
//
// Reading the pod's own token to discover the issuer is an UNVERIFIED,
// payload-only parse used solely for configuration. The token is never trusted
// on the strength of that parse: the issuer it yields is validated end to end
// (go-oidc matches it against the discovery document, and every request's token
// is verified against the discovered keys with its "iss" checked). The read
// pulls in no client-go dependency — it is a direct file read plus a stdlib
// base64/JSON decode.
//
// PROVISIONAL audience: the derived audience is the webhook's own
// service-account token audience, a placeholder until KEP-6060 finalizes the
// issuer-side audience derivation. Deployments that know their expected audience
// should use [NewRemoteVerifier] with that value.
func InCluster(ctx context.Context) (*verify.Verifier, error) {
	claims, err := readServiceAccountTokenClaims(serviceAccountTokenPath)
	if err != nil {
		return nil, fmt.Errorf("oidc: resolving in-cluster defaults: %w", err)
	}

	issuer := claims.Issuer
	if issuer == "" {
		return nil, errors.New("oidc: the projected service-account token has no \"iss\" claim; cannot determine the in-cluster issuer")
	}
	if len(claims.Audience) == 0 {
		return nil, errors.New("oidc: the projected service-account token has no \"aud\" claim; cannot derive a provisional in-cluster audience")
	}
	// PROVISIONAL: use the token's first audience as the expected audience. This
	// is one of two independent, opt-in provisional audience sources in this
	// module: this token-derived value (in-cluster) and the request/URL-derived
	// value in the admissionhttp package. They are separate entry points, not
	// alternative spellings of one another, and both must ultimately reconcile
	// to the same apiserver-side audience derivation once KEP-6060 finalizes it.
	audience := claims.Audience[0]

	client, err := clusterCAHTTPClient(serviceAccountCACertPath)
	if err != nil {
		return nil, fmt.Errorf("oidc: building in-cluster HTTP client: %w", err)
	}

	return NewRemoteVerifier(ctx, issuer, audience, WithHTTPClient(client))
}

// clusterCAHTTPClient builds an *http.Client whose transport trusts only the
// cluster CA bundle read from caCertPath, so OIDC discovery and JWKS fetches to
// the in-cluster issuer succeed over TLS without trusting the host roots.
func clusterCAHTTPClient(caCertPath string) (*http.Client, error) {
	pem, err := os.ReadFile(caCertPath)
	if err != nil {
		return nil, fmt.Errorf("reading cluster CA %q: %w", caCertPath, err)
	}
	pool := x509.NewCertPool()
	if !pool.AppendCertsFromPEM(pem) {
		return nil, fmt.Errorf("cluster CA %q contained no valid certificates", caCertPath)
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

// serviceAccountTokenClaims is the minimal, UNVERIFIED view of a JWT payload used
// only to discover the in-cluster issuer and a provisional audience. The token
// is never trusted on the strength of this parse: signatures and all claims are
// re-verified per request by the core Verifier against the discovered keys.
type serviceAccountTokenClaims struct {
	Issuer   string             `json:"iss"`
	Audience stringOrStringList `json:"aud"`
}

// stringOrStringList decodes a JWT claim that may be encoded either as a single
// string or as a list of strings (RFC 7519 §4.1.3). It exists only for the
// unverified service-account-token parse above; the verified request path uses
// go-oidc's own audience handling.
type stringOrStringList []string

func (a *stringOrStringList) UnmarshalJSON(b []byte) error {
	var list []string
	if err := json.Unmarshal(b, &list); err == nil {
		*a = list
		return nil
	}
	var single string
	if err := json.Unmarshal(b, &single); err != nil {
		return err
	}
	*a = stringOrStringList{single}
	return nil
}

// readServiceAccountTokenClaims reads the projected SA token at path and returns
// its UNVERIFIED "iss"/"aud" claims. It performs no signature check — it exists
// only to discover configuration defaults, and the returned issuer/audience are
// re-validated for every request by the core Verifier.
func readServiceAccountTokenClaims(path string) (serviceAccountTokenClaims, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return serviceAccountTokenClaims{}, fmt.Errorf("reading service-account token %q: %w", path, err)
	}
	return parseUnverifiedClaims(strings.TrimSpace(string(raw)))
}

// parseUnverifiedClaims decodes the payload segment of a compact JWS WITHOUT
// verifying its signature. Used solely to read defaults; never for trust.
func parseUnverifiedClaims(token string) (serviceAccountTokenClaims, error) {
	parts := strings.Split(token, ".")
	if len(parts) < 2 {
		return serviceAccountTokenClaims{}, errors.New("service-account token is not a JWT")
	}
	payload, err := base64.RawURLEncoding.DecodeString(parts[1])
	if err != nil {
		return serviceAccountTokenClaims{}, fmt.Errorf("decoding service-account token payload: %w", err)
	}
	var claims serviceAccountTokenClaims
	if err := json.Unmarshal(payload, &claims); err != nil {
		return serviceAccountTokenClaims{}, fmt.Errorf("parsing service-account token payload: %w", err)
	}
	return claims, nil
}
