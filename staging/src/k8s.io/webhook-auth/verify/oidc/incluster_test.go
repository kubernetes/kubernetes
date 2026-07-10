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

package oidc_test

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"encoding/pem"
	"os"
	"path/filepath"
	"testing"

	"k8s.io/webhook-auth/verify/oidc"
)

// writeFakeSAToken writes an UNSIGNED compact-JWT-shaped file whose payload
// carries the given claims. InCluster reads only the payload segment
// (unverified) to discover defaults, so no real signature is required.
func writeFakeSAToken(t *testing.T, dir string, claims map[string]any) string {
	t.Helper()
	payload, err := json.Marshal(claims)
	if err != nil {
		t.Fatalf("marshaling SA claims: %v", err)
	}
	header := base64.RawURLEncoding.EncodeToString([]byte(`{"alg":"RS256","typ":"JWT"}`))
	body := base64.RawURLEncoding.EncodeToString(payload)
	token := header + "." + body + ".c2ln" // "sig" — never checked

	path := filepath.Join(dir, "token")
	if err := os.WriteFile(path, []byte(token), 0o600); err != nil {
		t.Fatalf("writing SA token: %v", err)
	}
	return path
}

// writeServerCA writes the test server's TLS certificate as a PEM CA bundle,
// mirroring the projected ca.crt an in-cluster pod would trust.
func writeServerCA(t *testing.T, dir string, ts *oidcTestServer) string {
	t.Helper()
	cert := ts.server.Certificate()
	pemBytes := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: cert.Raw})
	path := filepath.Join(dir, "ca.crt")
	if err := os.WriteFile(path, pemBytes, 0o600); err != nil {
		t.Fatalf("writing CA bundle: %v", err)
	}
	return path
}

// TestInCluster_EndToEnd stands up a TLS OIDC server, projects a matching SA
// token (iss/aud) and CA bundle into temp files, and confirms InCluster wires a
// working verifier: it discovers the issuer, trusts the cluster CA, and verifies
// a real RS256 token minted by that server against the derived audience.
func TestInCluster_EndToEnd(t *testing.T) {
	ts := newOIDCTestServer(t)
	dir := t.TempDir()

	tokenPath := writeFakeSAToken(t, dir, map[string]any{
		"iss": ts.issuer,
		"aud": []string{testAudience},
	})
	caPath := writeServerCA(t, dir, ts)

	restore := oidc.SetInClusterPathsForTest(tokenPath, caPath)
	defer restore()

	v, err := oidc.InCluster(context.Background())
	if err != nil {
		t.Fatalf("InCluster: %v", err)
	}

	// A real KEP-6060 token minted by the server must verify end to end. The
	// provisional in-cluster audience is the SA token's aud (testAudience), which
	// baseClaims() also stamps into the token.
	res, err := v.Verify(context.Background(), ts.sign(t, ts.baseClaims()), testAPIGroup)
	if err != nil {
		t.Fatalf("verifying in-cluster token: %v", err)
	}
	if res.Issuer != ts.issuer {
		t.Errorf("issuer = %q, want %q", res.Issuer, ts.issuer)
	}
	if len(res.Audience) == 0 || res.Audience[0] != testAudience {
		t.Errorf("audience = %v, want %q", res.Audience, testAudience)
	}
}

// TestInCluster_SingleStringAudience confirms the unverified SA-token parse
// accepts the RFC 7519 single-string "aud" form.
func TestInCluster_SingleStringAudience(t *testing.T) {
	ts := newOIDCTestServer(t)
	dir := t.TempDir()

	tokenPath := writeFakeSAToken(t, dir, map[string]any{
		"iss": ts.issuer,
		"aud": testAudience, // single string, not an array
	})
	caPath := writeServerCA(t, dir, ts)

	restore := oidc.SetInClusterPathsForTest(tokenPath, caPath)
	defer restore()

	v, err := oidc.InCluster(context.Background())
	if err != nil {
		t.Fatalf("InCluster: %v", err)
	}
	if _, err := v.Verify(context.Background(), ts.sign(t, ts.baseClaims()), testAPIGroup); err != nil {
		t.Fatalf("verifying in-cluster token: %v", err)
	}
}

func TestInCluster_MissingConfig(t *testing.T) {
	tests := []struct {
		name   string
		claims map[string]any
	}{
		{
			name:   "SA token without iss fails",
			claims: map[string]any{"aud": []string{testAudience}},
		},
		{
			name:   "SA token without aud fails",
			claims: map[string]any{"iss": "https://issuer.example"},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			tokenPath := writeFakeSAToken(t, dir, tc.claims)
			caPath := filepath.Join(dir, "ca.crt")
			if err := os.WriteFile(caPath, []byte("not-a-cert"), 0o600); err != nil {
				t.Fatalf("writing CA: %v", err)
			}
			restore := oidc.SetInClusterPathsForTest(tokenPath, caPath)
			defer restore()

			if _, err := oidc.InCluster(context.Background()); err == nil {
				t.Fatal("expected error for incomplete in-cluster config")
			}
		})
	}
}

// TestInCluster_MissingTokenFile confirms a clear error when the projected SA
// token file is absent.
func TestInCluster_MissingTokenFile(t *testing.T) {
	dir := t.TempDir()
	restore := oidc.SetInClusterPathsForTest(filepath.Join(dir, "absent"), filepath.Join(dir, "ca.crt"))
	defer restore()

	if _, err := oidc.InCluster(context.Background()); err == nil {
		t.Fatal("expected error when the SA token file is missing")
	}
}

// TestInCluster_InvalidCA confirms a bad CA bundle fails construction rather
// than silently falling back to host roots.
func TestInCluster_InvalidCA(t *testing.T) {
	ts := newOIDCTestServer(t)
	dir := t.TempDir()

	tokenPath := writeFakeSAToken(t, dir, map[string]any{"iss": ts.issuer, "aud": []string{testAudience}})
	caPath := filepath.Join(dir, "ca.crt")
	if err := os.WriteFile(caPath, []byte("-----BEGIN CERTIFICATE-----\nnope\n-----END CERTIFICATE-----\n"), 0o600); err != nil {
		t.Fatalf("writing CA: %v", err)
	}
	restore := oidc.SetInClusterPathsForTest(tokenPath, caPath)
	defer restore()

	if _, err := oidc.InCluster(context.Background()); err == nil {
		t.Fatal("expected error for a CA bundle with no valid certificates")
	}
}
