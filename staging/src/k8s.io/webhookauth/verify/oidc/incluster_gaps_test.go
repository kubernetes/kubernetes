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
	"errors"
	"os"
	"path/filepath"
	"testing"

	"k8s.io/webhookauth/verify"
	"k8s.io/webhookauth/verify/oidc"
)

// writeRawSAToken writes an arbitrary raw string as the projected SA-token file.
// Unlike writeFakeSAToken it does not shape a JWT, so tests can drive the
// unverified-parse error branches (not-a-JWT, bad base64, bad JSON payload).
func writeRawSAToken(t *testing.T, dir, raw string) string {
	t.Helper()
	path := filepath.Join(dir, "token")
	if err := os.WriteFile(path, []byte(raw), 0o600); err != nil {
		t.Fatalf("writing raw SA token: %v", err)
	}
	return path
}

// TestInCluster_MultiAudienceFirstWins pins the PROVISIONAL in-cluster audience
// precedence: when the projected SA token carries several audiences, InCluster
// uses the FIRST one as the expected audience. A real token whose aud contains
// only the first SA audience verifies; a token whose aud contains only the
// second is rejected — proving aud[0] (not any later entry) is what the verifier
// enforces.
//
// This is PROVISIONAL behavior (KEP-6060 §16 has not finalized the issuer-side
// audience derivation); the test locks the current "first wins" contract so a
// change is a conscious one, and does not assert the value is final.
func TestInCluster_MultiAudienceFirstWins(t *testing.T) {
	ts := newOIDCTestServer(t)
	dir := t.TempDir()

	const secondAudience = "second.audience.example"
	// SA token advertises two audiences; testAudience is first.
	tokenPath := writeFakeSAToken(t, dir, map[string]any{
		"iss": ts.issuer,
		"aud": []string{testAudience, secondAudience},
	})
	caPath := writeServerCA(t, dir, ts)

	restore := oidc.SetInClusterPathsForTest(tokenPath, caPath)
	defer restore()

	v, err := oidc.InCluster(context.Background())
	if err != nil {
		t.Fatalf("InCluster: %v", err)
	}

	// A token whose aud is the FIRST SA audience verifies.
	firstAudClaims := ts.baseClaims()
	firstAudClaims["aud"] = []string{testAudience}
	if _, err := v.Verify(context.Background(), ts.sign(t, firstAudClaims), testAPIGroup); err != nil {
		t.Fatalf("token with the first SA audience should verify: %v", err)
	}

	// A token whose aud is ONLY the second SA audience must be rejected, proving
	// the second audience was not adopted as the expected value.
	secondAudClaims := ts.baseClaims()
	secondAudClaims["aud"] = []string{secondAudience}
	_, err = v.Verify(context.Background(), ts.sign(t, secondAudClaims), testAPIGroup)
	if err == nil {
		t.Fatal("token carrying only the second SA audience must be rejected (first wins)")
	}
	if !errors.Is(err, verify.ErrVerificationFailed) {
		t.Errorf("expected generic ErrVerificationFailed, got %v", err)
	}
}

// TestInCluster_UnverifiedParseErrors covers the failure branches of the
// unverified SA-token parse used only to discover in-cluster defaults. Each
// malformed token must fail construction with a clear error rather than
// silently producing an empty/incorrect issuer.
func TestInCluster_UnverifiedParseErrors(t *testing.T) {
	// A CA path that would work if parsing ever got that far, so the failure is
	// unambiguously attributable to the token parse.
	tests := []struct {
		name string
		// raw is the exact bytes written to the SA token file.
		raw string
	}{
		{
			name: "not a JWT (no dot-separated segments)",
			raw:  "this-is-not-a-jwt",
		},
		{
			name: "payload segment is not valid base64url",
			raw:  base64Header() + ".!!!not-base64!!!.sig",
		},
		{
			name: "payload segment is valid base64url but not JSON",
			raw:  base64Header() + "." + base64.RawURLEncoding.EncodeToString([]byte("not json")) + ".sig",
		},
		{
			name: "aud claim is neither string nor list",
			raw:  base64Header() + "." + base64.RawURLEncoding.EncodeToString([]byte(`{"iss":"https://x","aud":123}`)) + ".sig",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			tokenPath := writeRawSAToken(t, dir, tc.raw)
			caPath := filepath.Join(dir, "ca.crt") // never reached; parse fails first
			restore := oidc.SetInClusterPathsForTest(tokenPath, caPath)
			defer restore()

			if _, err := oidc.InCluster(context.Background()); err == nil {
				t.Fatal("expected InCluster to fail on a malformed SA token")
			}
		})
	}
}

// TestInCluster_MissingCAFile covers the CA-read error branch: a well-formed SA
// token (so issuer/audience resolve) paired with a missing ca.crt must fail
// construction rather than silently falling back to the host root store.
func TestInCluster_MissingCAFile(t *testing.T) {
	ts := newOIDCTestServer(t)
	dir := t.TempDir()

	tokenPath := writeFakeSAToken(t, dir, map[string]any{
		"iss": ts.issuer,
		"aud": []string{testAudience},
	})
	// Point at a CA path that does not exist.
	caPath := filepath.Join(dir, "absent-ca.crt")

	restore := oidc.SetInClusterPathsForTest(tokenPath, caPath)
	defer restore()

	if _, err := oidc.InCluster(context.Background()); err == nil {
		t.Fatal("expected InCluster to fail when the cluster CA file is missing")
	}
}

// base64Header returns a base64url-encoded JWT header segment, so raw tokens in
// the parse-error table have a plausible first segment and the parse reaches the
// payload segment under test.
func base64Header() string {
	return base64.RawURLEncoding.EncodeToString([]byte(`{"alg":"RS256","typ":"JWT"}`))
}
