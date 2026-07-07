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

package oidckeyset_test

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"k8s.io/webhook-auth/verify/oidckeyset"
)

// fakeInClusterKeySet is a verify.KeySet that never verifies anything; the
// in-cluster constructor only needs a non-nil key set to reach NewVerifier, so
// these tests inject it via WithKeySet and never call VerifySignature.
type fakeInClusterKeySet struct{}

func (fakeInClusterKeySet) VerifySignature(_ context.Context, raw string) ([]byte, error) {
	return []byte(raw), nil
}

// writeFakeSAToken writes an UNSIGNED compact-JWT-shaped file whose payload
// carries the given claims. NewInClusterVerifier reads only the payload segment
// (unverified) to discover defaults, so no real signature is required.
func writeFakeSAToken(t *testing.T, claims map[string]any) string {
	t.Helper()
	payload, err := json.Marshal(claims)
	if err != nil {
		t.Fatalf("marshaling SA claims: %v", err)
	}
	header := base64.RawURLEncoding.EncodeToString([]byte(`{"alg":"RS256","typ":"JWT"}`))
	body := base64.RawURLEncoding.EncodeToString(payload)
	token := header + "." + body + ".c2ln" // "sig" — never checked

	path := filepath.Join(t.TempDir(), "token")
	if err := os.WriteFile(path, []byte(token), 0o600); err != nil {
		t.Fatalf("writing SA token: %v", err)
	}
	return path
}

func TestNewInClusterVerifier_Defaults(t *testing.T) {
	const (
		issuer = "https://kubernetes.default.svc.cluster.local"
		saAud  = "https://kubernetes.default.svc"
	)

	tests := []struct {
		name    string
		claims  map[string]any
		opts    []oidckeyset.InClusterOption
		wantErr bool
	}{
		{
			name:   "zero-config: issuer and provisional audience from the SA token",
			claims: map[string]any{"iss": issuer, "aud": []string{saAud}},
		},
		{
			name:   "single-string aud in the SA token is accepted",
			claims: map[string]any{"iss": issuer, "aud": saAud},
		},
		{
			name:    "SA token without iss and no WithIssuer fails",
			claims:  map[string]any{"aud": []string{saAud}},
			wantErr: true,
		},
		{
			name:    "SA token without aud and no audience override fails",
			claims:  map[string]any{"iss": issuer},
			wantErr: true,
		},
		{
			name:   "WithServiceURL derives the provisional audience",
			claims: map[string]any{"iss": issuer},
			opts:   []oidckeyset.InClusterOption{oidckeyset.WithServiceURL("https://webhook.example.svc:443/validate")},
		},
		{
			name:   "WithAudiences overrides inference",
			claims: map[string]any{"iss": issuer, "aud": []string{saAud}},
			opts:   []oidckeyset.InClusterOption{oidckeyset.WithAudiences("https://explicit.example/aud")},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			tokenPath := writeFakeSAToken(t, tc.claims)
			opts := append([]oidckeyset.InClusterOption{
				oidckeyset.WithKeySet(fakeInClusterKeySet{}),
				oidckeyset.WithServiceAccountTokenPath(tokenPath),
			}, tc.opts...)

			v, err := oidckeyset.NewInClusterVerifier(context.Background(), opts...)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("expected error, got verifier %+v", v)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if v == nil {
				t.Fatal("expected a non-nil verifier")
			}
		})
	}
}

// TestNewInClusterVerifier_ExplicitConfigNoTokenRead confirms that when issuer,
// audiences, and key set are all supplied, the constructor works OFF-cluster —
// it must not require the projected SA token file.
func TestNewInClusterVerifier_ExplicitConfigNoTokenRead(t *testing.T) {
	v, err := oidckeyset.NewInClusterVerifier(
		context.Background(),
		oidckeyset.WithKeySet(fakeInClusterKeySet{}),
		oidckeyset.WithIssuer("https://issuer.example"),
		oidckeyset.WithAudiences("https://aud.example"),
		oidckeyset.WithServiceAccountTokenPath("/nonexistent/definitely/not/here"),
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if v == nil {
		t.Fatal("expected a non-nil verifier")
	}
}

// TestNewInClusterVerifier_MissingTokenFile confirms a clear error when a default
// needs the SA token but the file is absent.
func TestNewInClusterVerifier_MissingTokenFile(t *testing.T) {
	_, err := oidckeyset.NewInClusterVerifier(
		context.Background(),
		oidckeyset.WithKeySet(fakeInClusterKeySet{}),
		oidckeyset.WithServiceAccountTokenPath(filepath.Join(t.TempDir(), "absent")),
	)
	if err == nil {
		t.Fatal("expected error when the SA token file is missing")
	}
}
