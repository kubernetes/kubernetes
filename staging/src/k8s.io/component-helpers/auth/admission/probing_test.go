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

package admission_test

import (
	"bytes"
	"context"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	jose "github.com/go-jose/go-jose/v4"

	admissionv1 "k8s.io/api/admission/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/component-helpers/auth/admission"
	"k8s.io/component-helpers/auth/admission/internal/verify"
)

// ecdsaOIDCIssuer mimics the kube-apiserver's default ECDSA P-256 ServiceAccount
// token issuer, serving /.well-known/openid-configuration and /openid/v1/jwks
// while exposing its signing key so tests can mint both valid KEP-6060 tokens
// and adversarial tokens that kube-apiserver's TokenRequest API would refuse.
type ecdsaOIDCIssuer struct {
	server *httptest.Server
	issuer string
	signer jose.Signer
}

func newECDSAOIDCIssuer(t *testing.T) *ecdsaOIDCIssuer {
	t.Helper()
	priv, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatalf("generate ECDSA P-256 signing key: %v", err)
	}
	const keyID = "apiserver-ecdsa-key-1"
	signer, err := jose.NewSigner(
		jose.SigningKey{Algorithm: jose.ES256, Key: jose.JSONWebKey{Key: priv, KeyID: keyID}},
		(&jose.SignerOptions{}).WithType("JWT"),
	)
	if err != nil {
		t.Fatalf("create ES256 signer: %v", err)
	}
	jwks := jose.JSONWebKeySet{Keys: []jose.JSONWebKey{{
		Key:       priv.Public(),
		KeyID:     keyID,
		Algorithm: string(jose.ES256),
		Use:       "sig",
	}}}

	ts := &ecdsaOIDCIssuer{signer: signer}
	mux := http.NewServeMux()
	mux.HandleFunc("/.well-known/openid-configuration", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"issuer":                                ts.issuer,
			"jwks_uri":                              ts.issuer + "/openid/v1/jwks",
			"response_types_supported":              []string{"id_token"},
			"subject_types_supported":               []string{"public"},
			"id_token_signing_alg_values_supported": []string{string(jose.ES256), string(jose.RS256)},
		})
	})
	mux.HandleFunc("/openid/v1/jwks", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(jwks)
	})
	ts.server = httptest.NewTLSServer(mux)
	ts.issuer = ts.server.URL
	t.Cleanup(ts.server.Close)
	return ts
}

func (ts *ecdsaOIDCIssuer) sign(t *testing.T, claims map[string]any) string {
	t.Helper()
	payload, err := json.Marshal(claims)
	if err != nil {
		t.Fatalf("marshal claims: %v", err)
	}
	jws, err := ts.signer.Sign(payload)
	if err != nil {
		t.Fatalf("sign claims: %v", err)
	}
	compact, err := jws.CompactSerialize()
	if err != nil {
		t.Fatalf("serialize JWS: %v", err)
	}
	return compact
}

// validWebhookClaims builds the exact JWT claim shape minted by kube-apiserver's
// pkg/serviceaccount.Claims when APIServerWebhookAuthenticationToken is enabled.
func (ts *ecdsaOIDCIssuer) validWebhookClaims(audience, bindingKind, configName, configUID string, groups []string) map[string]any {
	now := time.Now()
	k8sClaims := map[string]any{
		"namespace": "kube-system",
		"serviceaccount": map[string]any{
			"name": "webhook-authenticator",
			"uid":  "sa-uid-1111",
		},
		"attestations": map[string]any{
			admissionReviewAPIGroupsClaimKey: groups,
		},
	}
	if bindingKind != "" {
		k8sClaims[bindingKind] = map[string]any{
			"name": configName,
			"uid":  configUID,
		}
	}
	return map[string]any{
		"iss":           ts.issuer,
		"sub":           "system:serviceaccount:kube-system:webhook-authenticator",
		"aud":           []string{audience},
		"exp":           now.Add(5 * time.Minute).Unix(),
		"nbf":           now.Add(-1 * time.Minute).Unix(),
		"iat":           now.Unix(),
		"jti":           "jti-webhook-token-1",
		"kubernetes.io": k8sClaims,
	}
}

// forgeAlgNoneJWT builds an unsigned compact JWT ("alg":"none") for probing.
func forgeAlgNoneJWT(t *testing.T, claims map[string]any) string {
	t.Helper()
	hdr := base64.RawURLEncoding.EncodeToString([]byte(`{"alg":"none","typ":"JWT"}`))
	payloadBytes, err := json.Marshal(claims)
	if err != nil {
		t.Fatalf("marshal alg:none claims: %v", err)
	}
	payload := base64.RawURLEncoding.EncodeToString(payloadBytes)
	return hdr + "." + payload + "."
}

// forgeHS256JWT builds a symmetric HS256 JWT for algorithm-confusion probing.
func forgeHS256JWT(t *testing.T, claims map[string]any) string {
	t.Helper()
	hdr := base64.RawURLEncoding.EncodeToString([]byte(`{"alg":"HS256","kid":"apiserver-ecdsa-key-1","typ":"JWT"}`))
	payloadBytes, err := json.Marshal(claims)
	if err != nil {
		t.Fatalf("marshal HS256 claims: %v", err)
	}
	payload := base64.RawURLEncoding.EncodeToString(payloadBytes)
	mac := hmac.New(sha256.New, []byte("probing-symmetric-secret"))
	mac.Write([]byte(hdr + "." + payload))
	sig := base64.RawURLEncoding.EncodeToString(mac.Sum(nil))
	return hdr + "." + payload + "." + sig
}

// TestDummyWebhookEndToEndAndAntiProbing stands up a realistic dummy webhook
// server exposing both /validate and /mutate endpoints in both in-cluster
// discovery mode and remote-issuer mode. It verifies that:
//  1. Legitimate KEP-6060 tokens succeed for exact API groups, wildcard "*",
//     and the three allowed cross-group subresources (Scale, Eviction, TokenRequest).
//  2. Every pre-decode probe (missing/malformed auth header, wrong HTTP verb,
//     empty/malformed/oversized body) returns the exact same 401 response byte-for-byte.
//  3. Every post-decode probe (forged key, alg:none, HS256, wrong issuer/audience,
//     spoofed Host header, expired/future token, Pod/Secret/Node-bound SA token,
//     dual or incomplete webhook binding, missing/empty/multi-group attestation,
//     unauthorized API group, subresource group spoofing, Object/OldObject GVK mismatch)
//     returns the exact same HTTP 200 AdmissionReview denial byte-for-byte, so an
//     attacker cannot use the webhook as an oracle to probe audiences, groups, or claims.
func TestDummyWebhookEndToEndAndAntiProbing(t *testing.T) {
	issuer := newECDSAOIDCIssuer(t)
	otherIssuer := newECDSAOIDCIssuer(t)

	validateAudience := verify.AudienceForService("dummy-webhook", "webhook-system", 443, "/validate")
	mutateAudience := verify.AudienceForService("dummy-webhook", "webhook-system", 443, "/mutate")

	modes := []struct {
		name        string
		makeOptions func(audience string) []admission.Option
	}{
		{
			name: "in-cluster-discovery",
			makeOptions: func(audience string) []admission.Option {
				return []admission.Option{
					admission.WithInClusterEndpointForTest(issuer.issuer, issuer.server.Client()),
					admission.WithAudience(audience),
					admission.WithMaxBodyBytesForTest(4096),
				}
			},
		},
		{
			name: "remote-issuer",
			makeOptions: func(audience string) []admission.Option {
				return []admission.Option{
					admission.WithRemoteIssuer(issuer.issuer, audience, issuer.server.Client()),
					admission.WithMaxBodyBytesForTest(4096),
				}
			},
		},
	}

	for _, mode := range modes {
		t.Run(mode.name, func(t *testing.T) {
			var validateCalls, mutateCalls atomic.Int32

			validateHandler, err := admission.WithTokenVerification(
				context.Background(),
				func(_ context.Context, req *admissionv1.AdmissionRequest) *admissionv1.AdmissionResponse {
					validateCalls.Add(1)
					return &admissionv1.AdmissionResponse{
						Allowed: true,
						Result:  &metav1.Status{Status: metav1.StatusSuccess, Message: "validated:" + req.Resource.Resource},
					}
				},
				mode.makeOptions(validateAudience)...,
			)
			if err != nil {
				t.Fatalf("construct /validate handler: %v", err)
			}

			patchType := admissionv1.PatchTypeJSONPatch
			mutateHandler, err := admission.WithTokenVerification(
				context.Background(),
				func(_ context.Context, _ *admissionv1.AdmissionRequest) *admissionv1.AdmissionResponse {
					mutateCalls.Add(1)
					return &admissionv1.AdmissionResponse{
						Allowed:   true,
						PatchType: &patchType,
						Patch:     []byte(`[{"op":"add","path":"/metadata/annotations","value":{"kep6060":"verified"}}]`),
					}
				},
				mode.makeOptions(mutateAudience)...,
			)
			if err != nil {
				t.Fatalf("construct /mutate handler: %v", err)
			}

			mux := http.NewServeMux()
			mux.Handle("/validate", validateHandler)
			mux.Handle("/mutate", mutateHandler)
			webhookServer := httptest.NewTLSServer(mux)
			t.Cleanup(webhookServer.Close)

			appsReview := func(uid types.UID) []byte {
				b, err := json.Marshal(&admissionv1.AdmissionReview{
					TypeMeta: metav1.TypeMeta{APIVersion: "admission.k8s.io/v1", Kind: "AdmissionReview"},
					Request: &admissionv1.AdmissionRequest{
						UID:       uid,
						Kind:      metav1.GroupVersionKind{Group: "apps", Version: "v1", Kind: "Deployment"},
						Resource:  metav1.GroupVersionResource{Group: "apps", Version: "v1", Resource: "deployments"},
						Operation: admissionv1.Create,
						Object:    runtime.RawExtension{Raw: []byte(`{"apiVersion":"apps/v1","kind":"Deployment","metadata":{"name":"demo"}}`)},
					},
				})
				if err != nil {
					t.Fatalf("marshal apps review: %v", err)
				}
				return b
			}

			validToken := issuer.sign(t, issuer.validWebhookClaims(
				validateAudience,
				"validatingwebhookconfiguration",
				"dummy-validating-cfg",
				"vwc-uid-1234",
				[]string{"apps"},
			))

			// 1. Pre-decode probing matrix: every probe must return HTTP 401 with
			// the exact same response body and never reach the handler.
			preDecodeProbes := []struct {
				name       string
				method     string
				authHeader string
				body       []byte
			}{
				{
					name:   "missing Authorization header",
					method: http.MethodPost,
					body:   appsReview("probe-1"),
				},
				{
					name:       "Basic auth scheme instead of Bearer",
					method:     http.MethodPost,
					authHeader: "Basic dXNlcjpwYXNz",
					body:       appsReview("probe-2"),
				},
				{
					name:       "Bearer with empty token",
					method:     http.MethodPost,
					authHeader: "Bearer ",
					body:       appsReview("probe-3"),
				},
				{
					name:       "Bearer with extra leading space",
					method:     http.MethodPost,
					authHeader: "Bearer  " + validToken,
					body:       appsReview("probe-4"),
				},
				{
					name:       "GET probe with valid token and no body",
					method:     http.MethodGet,
					authHeader: "Bearer " + validToken,
					body:       nil,
				},
				{
					name:       "DELETE probe with valid token and no body",
					method:     http.MethodDelete,
					authHeader: "Bearer " + validToken,
					body:       nil,
				},
				{
					name:       "malformed JSON body with valid token",
					method:     http.MethodPost,
					authHeader: "Bearer " + validToken,
					body:       []byte(`{"not":"an admission review"`),
				},
				{
					name:       "AdmissionReview with nil Request and valid token",
					method:     http.MethodPost,
					authHeader: "Bearer " + validToken,
					body:       []byte(`{"apiVersion":"admission.k8s.io/v1","kind":"AdmissionReview"}`),
				},
				{
					name:       "oversized body exceeding maxBodyBytes with valid token",
					method:     http.MethodPost,
					authHeader: "Bearer " + validToken,
					body:       []byte(`{"apiVersion":"admission.k8s.io/v1","kind":"AdmissionReview","request":{"uid":"big","object":{"apiVersion":"apps/v1","kind":"Deployment","pad":"` + strings.Repeat("A", 5000) + `"}}}`),
				},
			}

			var canonical401Body string
			for i, probe := range preDecodeProbes {
				t.Run("pre-decode/"+probe.name, func(t *testing.T) {
					before := validateCalls.Load()
					var bodyReader io.Reader
					if probe.body != nil {
						bodyReader = bytes.NewReader(probe.body)
					}
					req, err := http.NewRequestWithContext(context.Background(), probe.method, webhookServer.URL+"/validate", bodyReader)
					if err != nil {
						t.Fatalf("new request: %v", err)
					}
					req.Header.Set("Content-Type", "application/json")
					if probe.authHeader != "" {
						req.Header.Set("Authorization", probe.authHeader)
					}
					resp, err := webhookServer.Client().Do(req)
					if err != nil {
						t.Fatalf("execute request: %v", err)
					}
					defer resp.Body.Close()
					rawBytes, err := io.ReadAll(resp.Body)
					if err != nil {
						t.Fatalf("read response body: %v", err)
					}
					if resp.StatusCode != http.StatusUnauthorized {
						t.Fatalf("status = %d, want 401; body = %q", resp.StatusCode, string(rawBytes))
					}
					if validateCalls.Load() != before {
						t.Fatalf("downstream handler was invoked on pre-decode rejection")
					}
					if string(rawBytes) != "webhook token verification failed\n" {
						t.Fatalf("body = %q, want generic 401 message", string(rawBytes))
					}
					if i == 0 {
						canonical401Body = string(rawBytes)
					} else if string(rawBytes) != canonical401Body {
						t.Fatalf("pre-decode response body %q differs from canonical %q (oracle leak)", string(rawBytes), canonical401Body)
					}
				})
			}

			// 2. Post-decode probing matrix: every probe must return HTTP 200 with
			// a byte-for-byte identical AdmissionReview denial (for a constant UID)
			// and never invoke the downstream handler.
			const fixedProbeUID = types.UID("anti-oracle-probe-uid")
			defaultAppsRequest := func() *admissionv1.AdmissionRequest {
				return &admissionv1.AdmissionRequest{
					UID:       fixedProbeUID,
					Kind:      metav1.GroupVersionKind{Group: "apps", Version: "v1", Kind: "Deployment"},
					Resource:  metav1.GroupVersionResource{Group: "apps", Version: "v1", Resource: "deployments"},
					Operation: admissionv1.Create,
					Object:    runtime.RawExtension{Raw: []byte(`{"apiVersion":"apps/v1","kind":"Deployment","metadata":{"name":"demo"}}`)},
				}
			}

			postDecodeProbes := []struct {
				name  string
				host  string
				token func() string
				req   func() *admissionv1.AdmissionRequest
			}{
				{
					name:  "non-JWT opaque token",
					token: func() string { return "not-a-jwt-token" },
					req:   defaultAppsRequest,
				},
				{
					name: "unsigned alg:none JWT",
					token: func() string {
						return forgeAlgNoneJWT(t, issuer.validWebhookClaims(validateAudience, "validatingwebhookconfiguration", "vwc", "uid-1", []string{"apps"}))
					},
					req: defaultAppsRequest,
				},
				{
					name: "symmetric HS256 algorithm confusion JWT",
					token: func() string {
						return forgeHS256JWT(t, issuer.validWebhookClaims(validateAudience, "validatingwebhookconfiguration", "vwc", "uid-1", []string{"apps"}))
					},
					req: defaultAppsRequest,
				},
				{
					name: "signed by untrusted key with same issuer claim",
					token: func() string {
						c := issuer.validWebhookClaims(validateAudience, "validatingwebhookconfiguration", "vwc", "uid-1", []string{"apps"})
						return otherIssuer.sign(t, c)
					},
					req: defaultAppsRequest,
				},
				{
					name: "untrusted issuer claim",
					token: func() string {
						c := issuer.validWebhookClaims(validateAudience, "validatingwebhookconfiguration", "vwc", "uid-1", []string{"apps"})
						c["iss"] = "https://evil-issuer.example.com"
						return issuer.sign(t, c)
					},
					req: defaultAppsRequest,
				},
				{
					name: "wrong audience (token minted for /mutate sent to /validate)",
					token: func() string {
						return issuer.sign(t, issuer.validWebhookClaims(mutateAudience, "mutatingwebhookconfiguration", "mwc", "uid-1", []string{"apps"}))
					},
					req: defaultAppsRequest,
				},
				{
					name: "spoofed HTTP Host header attempting to match wrong token audience",
					host: "other-webhook.webhook-system.svc:443",
					token: func() string {
						otherAud := verify.AudienceForService("other-webhook", "webhook-system", 443, "/validate")
						return issuer.sign(t, issuer.validWebhookClaims(otherAud, "validatingwebhookconfiguration", "vwc", "uid-1", []string{"apps"}))
					},
					req: defaultAppsRequest,
				},
				{
					name: "expired token",
					token: func() string {
						c := issuer.validWebhookClaims(validateAudience, "validatingwebhookconfiguration", "vwc", "uid-1", []string{"apps"})
						c["exp"] = time.Now().Add(-2 * time.Minute).Unix()
						return issuer.sign(t, c)
					},
					req: defaultAppsRequest,
				},
				{
					name: "not-before in the future",
					token: func() string {
						c := issuer.validWebhookClaims(validateAudience, "validatingwebhookconfiguration", "vwc", "uid-1", []string{"apps"})
						c["nbf"] = time.Now().Add(10 * time.Minute).Unix()
						return issuer.sign(t, c)
					},
					req: defaultAppsRequest,
				},
				{
					name: "ordinary Pod-bound ServiceAccount token",
					token: func() string {
						c := issuer.validWebhookClaims(validateAudience, "pod", "my-pod", "pod-uid-1", []string{"apps"})
						return issuer.sign(t, c)
					},
					req: defaultAppsRequest,
				},
				{
					name: "ordinary Secret-bound ServiceAccount token",
					token: func() string {
						c := issuer.validWebhookClaims(validateAudience, "secret", "my-secret", "sec-uid-1", []string{"apps"})
						return issuer.sign(t, c)
					},
					req: defaultAppsRequest,
				},
				{
					name: "ordinary Node-bound ServiceAccount token",
					token: func() string {
						c := issuer.validWebhookClaims(validateAudience, "node", "node-1", "node-uid-1", []string{"apps"})
						return issuer.sign(t, c)
					},
					req: defaultAppsRequest,
				},
				{
					name: "unbound ServiceAccount token (no webhook binding)",
					token: func() string {
						c := issuer.validWebhookClaims(validateAudience, "", "", "", []string{"apps"})
						return issuer.sign(t, c)
					},
					req: defaultAppsRequest,
				},
				{
					name: "token bound to both Validating and Mutating webhook configurations",
					token: func() string {
						c := issuer.validWebhookClaims(validateAudience, "validatingwebhookconfiguration", "vwc", "uid-1", []string{"apps"})
						c["kubernetes.io"].(map[string]any)["mutatingwebhookconfiguration"] = map[string]any{"name": "mwc", "uid": "uid-2"}
						return issuer.sign(t, c)
					},
					req: defaultAppsRequest,
				},
				{
					name: "webhook binding missing UID",
					token: func() string {
						return issuer.sign(t, issuer.validWebhookClaims(validateAudience, "validatingwebhookconfiguration", "vwc", "", []string{"apps"}))
					},
					req: defaultAppsRequest,
				},
				{
					name: "webhook binding missing Name",
					token: func() string {
						return issuer.sign(t, issuer.validWebhookClaims(validateAudience, "validatingwebhookconfiguration", "", "uid-1", []string{"apps"}))
					},
					req: defaultAppsRequest,
				},
				{
					name: "missing admissionReviewAPIGroups attestation",
					token: func() string {
						c := issuer.validWebhookClaims(validateAudience, "validatingwebhookconfiguration", "vwc", "uid-1", []string{"apps"})
						delete(c["kubernetes.io"].(map[string]any), "attestations")
						return issuer.sign(t, c)
					},
					req: defaultAppsRequest,
				},
				{
					name: "empty admissionReviewAPIGroups slice",
					token: func() string {
						return issuer.sign(t, issuer.validWebhookClaims(validateAudience, "validatingwebhookconfiguration", "vwc", "uid-1", []string{}))
					},
					req: defaultAppsRequest,
				},
				{
					name: "empty-string element in admissionReviewAPIGroups slice",
					token: func() string {
						return issuer.sign(t, issuer.validWebhookClaims(validateAudience, "validatingwebhookconfiguration", "vwc", "uid-1", []string{""}))
					},
					req: defaultAppsRequest,
				},
				{
					name: "multi-valued admissionReviewAPIGroups slice (disallowed by KEP-6060)",
					token: func() string {
						return issuer.sign(t, issuer.validWebhookClaims(validateAudience, "validatingwebhookconfiguration", "vwc", "uid-1", []string{"apps", "batch"}))
					},
					req: defaultAppsRequest,
				},
				{
					name:  "unauthorized API group (token authorizes apps, request is batch)",
					token: func() string { return validToken },
					req: func() *admissionv1.AdmissionRequest {
						return &admissionv1.AdmissionRequest{
							UID:       fixedProbeUID,
							Kind:      metav1.GroupVersionKind{Group: "batch", Version: "v1", Kind: "Job"},
							Resource:  metav1.GroupVersionResource{Group: "batch", Version: "v1", Resource: "jobs"},
							Operation: admissionv1.Create,
							Object:    runtime.RawExtension{Raw: []byte(`{"apiVersion":"batch/v1","kind":"Job"}`)},
						}
					},
				},
				{
					name:  "unauthorized core API group (token authorizes apps, request is core Pod)",
					token: func() string { return validToken },
					req: func() *admissionv1.AdmissionRequest {
						return &admissionv1.AdmissionRequest{
							UID:       fixedProbeUID,
							Kind:      metav1.GroupVersionKind{Group: "", Version: "v1", Kind: "Pod"},
							Resource:  metav1.GroupVersionResource{Group: "", Version: "v1", Resource: "pods"},
							Operation: admissionv1.Create,
							Object:    runtime.RawExtension{Raw: []byte(`{"apiVersion":"v1","kind":"Pod"}`)},
						}
					},
				},
				{
					name:  "subresource group spoofing (deployments/scale carrying rbac Role)",
					token: func() string { return validToken },
					req: func() *admissionv1.AdmissionRequest {
						return &admissionv1.AdmissionRequest{
							UID:         fixedProbeUID,
							Kind:        metav1.GroupVersionKind{Group: "rbac.authorization.k8s.io", Version: "v1", Kind: "Role"},
							Resource:    metav1.GroupVersionResource{Group: "apps", Version: "v1", Resource: "deployments"},
							SubResource: "scale",
							Operation:   admissionv1.Update,
							Object:      runtime.RawExtension{Raw: []byte(`{"apiVersion":"rbac.authorization.k8s.io/v1","kind":"Role"}`)},
						}
					},
				},
				{
					name:  "raw Object group mismatches AdmissionRequest.Resource.Group",
					token: func() string { return validToken },
					req: func() *admissionv1.AdmissionRequest {
						r := defaultAppsRequest()
						r.Object = runtime.RawExtension{Raw: []byte(`{"apiVersion":"v1","kind":"Secret","metadata":{"name":"stolen"}}`)}
						return r
					},
				},
				{
					name:  "raw OldObject group mismatches AdmissionRequest.Resource.Group",
					token: func() string { return validToken },
					req: func() *admissionv1.AdmissionRequest {
						r := defaultAppsRequest()
						r.Operation = admissionv1.Update
						r.OldObject = runtime.RawExtension{Raw: []byte(`{"apiVersion":"v1","kind":"Secret","metadata":{"name":"stolen"}}`)}
						return r
					},
				},
			}

			var canonicalDenialBytes []byte
			for i, probe := range postDecodeProbes {
				t.Run("post-decode/"+probe.name, func(t *testing.T) {
					before := validateCalls.Load()
					reviewBytes, err := json.Marshal(&admissionv1.AdmissionReview{
						TypeMeta: metav1.TypeMeta{APIVersion: "admission.k8s.io/v1", Kind: "AdmissionReview"},
						Request:  probe.req(),
					})
					if err != nil {
						t.Fatalf("marshal probe review: %v", err)
					}
					req, err := http.NewRequestWithContext(context.Background(), http.MethodPost, webhookServer.URL+"/validate", bytes.NewReader(reviewBytes))
					if err != nil {
						t.Fatalf("new request: %v", err)
					}
					if probe.host != "" {
						req.Host = probe.host
					}
					req.Header.Set("Content-Type", "application/json")
					req.Header.Set("Authorization", "Bearer "+probe.token())

					resp, err := webhookServer.Client().Do(req)
					if err != nil {
						t.Fatalf("execute request: %v", err)
					}
					defer resp.Body.Close()
					rawBytes, err := io.ReadAll(resp.Body)
					if err != nil {
						t.Fatalf("read response body: %v", err)
					}
					if resp.StatusCode != http.StatusOK {
						t.Fatalf("status = %d, want 200; body = %q", resp.StatusCode, string(rawBytes))
					}
					if validateCalls.Load() != before {
						t.Fatalf("downstream handler was invoked on post-decode rejection")
					}
					var decoded admissionv1.AdmissionReview
					if err := json.Unmarshal(rawBytes, &decoded); err != nil {
						t.Fatalf("decode AdmissionReview response: %v", err)
					}
					if decoded.Response == nil || decoded.Response.Allowed {
						t.Fatalf("expected explicit Allowed:false denial, got %+v", decoded.Response)
					}
					if decoded.Response.UID != fixedProbeUID {
						t.Fatalf("UID = %q, want %q", decoded.Response.UID, fixedProbeUID)
					}
					if decoded.Response.Result == nil || decoded.Response.Result.Code != http.StatusUnauthorized || decoded.Response.Result.Message != "webhook token verification failed" {
						t.Fatalf("unexpected denial status: %+v", decoded.Response.Result)
					}
					if i == 0 {
						canonicalDenialBytes = rawBytes
					} else if !bytes.Equal(rawBytes, canonicalDenialBytes) {
						t.Fatalf("post-decode denial bytes %q differ from canonical %q (oracle leak)", string(rawBytes), string(canonicalDenialBytes))
					}
				})
			}

			// 3. Positive exact-scope matrix: verify that valid tokens for both
			// validating and mutating webhooks succeed across named groups, core "",
			// wildcard "*", and the three legitimate cross-group subresources.
			positiveCases := []struct {
				name        string
				path        string
				audience    string
				bindingKind string
				attGroup    string
				req         *admissionv1.AdmissionRequest
				wantMutated bool
			}{
				{
					name:        "validating apps/v1 Deployment",
					path:        "/validate",
					audience:    validateAudience,
					bindingKind: "validatingwebhookconfiguration",
					attGroup:    "apps",
					req:         defaultAppsRequest(),
				},
				{
					name:        "mutating apps/v1 Deployment",
					path:        "/mutate",
					audience:    mutateAudience,
					bindingKind: "mutatingwebhookconfiguration",
					attGroup:    "apps",
					req:         defaultAppsRequest(),
					wantMutated: true,
				},
				{
					name:        "validating core v1 Pod with wildcard attestation",
					path:        "/validate",
					audience:    validateAudience,
					bindingKind: "validatingwebhookconfiguration",
					attGroup:    "*",
					req: &admissionv1.AdmissionRequest{
						UID:       "pos-core-pod",
						Kind:      metav1.GroupVersionKind{Group: "", Version: "v1", Kind: "Pod"},
						Resource:  metav1.GroupVersionResource{Group: "", Version: "v1", Resource: "pods"},
						Operation: admissionv1.Create,
						Object:    runtime.RawExtension{Raw: []byte(`{"apiVersion":"v1","kind":"Pod","metadata":{"name":"p1"}}`)},
					},
				},
				{
					name:        "validating deployments/scale cross-group subresource",
					path:        "/validate",
					audience:    validateAudience,
					bindingKind: "validatingwebhookconfiguration",
					attGroup:    "apps",
					req: &admissionv1.AdmissionRequest{
						UID:         "pos-scale",
						Kind:        metav1.GroupVersionKind{Group: "autoscaling", Version: "v1", Kind: "Scale"},
						Resource:    metav1.GroupVersionResource{Group: "apps", Version: "v1", Resource: "deployments"},
						SubResource: "scale",
						Operation:   admissionv1.Update,
						Object:      runtime.RawExtension{Raw: []byte(`{"apiVersion":"autoscaling/v1","kind":"Scale"}`)},
					},
				},
				{
					name:        "validating pods/eviction cross-group subresource",
					path:        "/validate",
					audience:    validateAudience,
					bindingKind: "validatingwebhookconfiguration",
					attGroup:    "*",
					req: &admissionv1.AdmissionRequest{
						UID:         "pos-eviction",
						Kind:        metav1.GroupVersionKind{Group: "policy", Version: "v1", Kind: "Eviction"},
						Resource:    metav1.GroupVersionResource{Group: "", Version: "v1", Resource: "pods"},
						SubResource: "eviction",
						Operation:   admissionv1.Create,
						Object:      runtime.RawExtension{Raw: []byte(`{"apiVersion":"policy/v1","kind":"Eviction"}`)},
					},
				},
				{
					name:        "validating serviceaccounts/token cross-group subresource",
					path:        "/validate",
					audience:    validateAudience,
					bindingKind: "validatingwebhookconfiguration",
					attGroup:    "*",
					req: &admissionv1.AdmissionRequest{
						UID:         "pos-satoken",
						Kind:        metav1.GroupVersionKind{Group: "authentication.k8s.io", Version: "v1", Kind: "TokenRequest"},
						Resource:    metav1.GroupVersionResource{Group: "", Version: "v1", Resource: "serviceaccounts"},
						SubResource: "token",
						Operation:   admissionv1.Create,
						Object:      runtime.RawExtension{Raw: []byte(`{"apiVersion":"authentication.k8s.io/v1","kind":"TokenRequest"}`)},
					},
				},
			}

			for _, tc := range positiveCases {
				if tc.req == nil {
					continue
				}
				t.Run("positive/"+tc.name, func(t *testing.T) {
					token := issuer.sign(t, issuer.validWebhookClaims(tc.audience, tc.bindingKind, "cfg-name", "cfg-uid", []string{tc.attGroup}))
					reviewBytes, err := json.Marshal(&admissionv1.AdmissionReview{
						TypeMeta: metav1.TypeMeta{APIVersion: "admission.k8s.io/v1", Kind: "AdmissionReview"},
						Request:  tc.req,
					})
					if err != nil {
						t.Fatalf("marshal positive review: %v", err)
					}
					req, err := http.NewRequestWithContext(context.Background(), http.MethodPost, webhookServer.URL+tc.path, bytes.NewReader(reviewBytes))
					if err != nil {
						t.Fatalf("new request: %v", err)
					}
					req.Header.Set("Content-Type", "application/json")
					req.Header.Set("Authorization", "Bearer "+token)

					resp, err := webhookServer.Client().Do(req)
					if err != nil {
						t.Fatalf("execute request: %v", err)
					}
					defer resp.Body.Close()
					if resp.StatusCode != http.StatusOK {
						t.Fatalf("status = %d, want 200", resp.StatusCode)
					}
					var decoded admissionv1.AdmissionReview
					if err := json.NewDecoder(resp.Body).Decode(&decoded); err != nil {
						t.Fatalf("decode response: %v", err)
					}
					if decoded.Response == nil || !decoded.Response.Allowed || decoded.Response.UID != tc.req.UID {
						t.Fatalf("unexpected positive response: %+v", decoded.Response)
					}
					if tc.wantMutated && len(decoded.Response.Patch) == 0 {
						t.Fatalf("expected JSONPatch on mutating response, got none")
					}
				})
			}
		})
	}
}

// TestTokenCacheIsolatesPerRequestChecks verifies that caching a verified token
// never bypasses per-request API-group or embedded-object checks on subsequent
// requests presenting that same token.
func TestTokenCacheIsolatesPerRequestChecks(t *testing.T) {
	issuer := newECDSAOIDCIssuer(t)
	audience := verify.AudienceForService("dummy-webhook", "webhook-system", 443, "/validate")

	var admitted atomic.Int32
	handler, err := admission.WithTokenVerification(
		context.Background(),
		func(_ context.Context, _ *admissionv1.AdmissionRequest) *admissionv1.AdmissionResponse {
			admitted.Add(1)
			return &admissionv1.AdmissionResponse{Allowed: true}
		},
		admission.WithRemoteIssuer(issuer.issuer, audience, issuer.server.Client()),
	)
	if err != nil {
		t.Fatalf("construct handler: %v", err)
	}
	srv := httptest.NewTLSServer(handler)
	t.Cleanup(srv.Close)

	token := issuer.sign(t, issuer.validWebhookClaims(audience, "validatingwebhookconfiguration", "vwc", "uid-1", []string{"apps"}))

	send := func(req *admissionv1.AdmissionRequest) *admissionv1.AdmissionResponse {
		t.Helper()
		b, err := json.Marshal(&admissionv1.AdmissionReview{
			TypeMeta: metav1.TypeMeta{APIVersion: "admission.k8s.io/v1", Kind: "AdmissionReview"},
			Request:  req,
		})
		if err != nil {
			t.Fatalf("marshal review: %v", err)
		}
		httpReq, err := http.NewRequestWithContext(context.Background(), http.MethodPost, srv.URL, bytes.NewReader(b))
		if err != nil {
			t.Fatalf("new request: %v", err)
		}
		httpReq.Header.Set("Content-Type", "application/json")
		httpReq.Header.Set("Authorization", "Bearer "+token)
		resp, err := srv.Client().Do(httpReq)
		if err != nil {
			t.Fatalf("do request: %v", err)
		}
		defer resp.Body.Close()
		var out admissionv1.AdmissionReview
		if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
			t.Fatalf("decode response: %v", err)
		}
		return out.Response
	}

	validApps := &admissionv1.AdmissionRequest{
		UID:       "req-1",
		Kind:      metav1.GroupVersionKind{Group: "apps", Version: "v1", Kind: "Deployment"},
		Resource:  metav1.GroupVersionResource{Group: "apps", Version: "v1", Resource: "deployments"},
		Operation: admissionv1.Create,
		Object:    runtime.RawExtension{Raw: []byte(`{"apiVersion":"apps/v1","kind":"Deployment"}`)},
	}

	// Warm the positive token cache with a valid apps/v1 request.
	if resp := send(validApps); resp == nil || !resp.Allowed || admitted.Load() != 1 {
		t.Fatalf("first valid request failed: %+v", resp)
	}

	// Present the exact same cached token on a batch/v1 request: must be denied.
	unauthorizedBatch := &admissionv1.AdmissionRequest{
		UID:       "req-2",
		Kind:      metav1.GroupVersionKind{Group: "batch", Version: "v1", Kind: "Job"},
		Resource:  metav1.GroupVersionResource{Group: "batch", Version: "v1", Resource: "jobs"},
		Operation: admissionv1.Create,
		Object:    runtime.RawExtension{Raw: []byte(`{"apiVersion":"batch/v1","kind":"Job"}`)},
	}
	if resp := send(unauthorizedBatch); resp == nil || resp.Allowed || admitted.Load() != 1 {
		t.Fatalf("cached token must not authorize batch/v1: %+v", resp)
	}

	// Present the exact same cached token on an apps/v1 request whose embedded
	// Object is a core v1 Secret: must be denied.
	spoofedObject := &admissionv1.AdmissionRequest{
		UID:       "req-3",
		Kind:      metav1.GroupVersionKind{Group: "apps", Version: "v1", Kind: "Deployment"},
		Resource:  metav1.GroupVersionResource{Group: "apps", Version: "v1", Resource: "deployments"},
		Operation: admissionv1.Create,
		Object:    runtime.RawExtension{Raw: []byte(`{"apiVersion":"v1","kind":"Secret"}`)},
	}
	if resp := send(spoofedObject); resp == nil || resp.Allowed || admitted.Load() != 1 {
		t.Fatalf("cached token must not allow spoofed Object GVK: %+v", resp)
	}

	// Follow-up valid apps/v1 request with the cached token still succeeds.
	validApps.UID = "req-4"
	if resp := send(validApps); resp == nil || !resp.Allowed || admitted.Load() != 2 {
		t.Fatalf("second valid request failed: %+v", resp)
	}
}
