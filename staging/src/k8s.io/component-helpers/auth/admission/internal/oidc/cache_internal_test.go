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
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	coreosoidc "github.com/coreos/go-oidc/v3/oidc"
)

type countingKeySet struct {
	calls atomic.Int32
}

func (k *countingKeySet) VerifySignature(_ context.Context, rawToken string) ([]byte, error) {
	k.calls.Add(1)
	return base64.RawURLEncoding.DecodeString(strings.Split(rawToken, ".")[1])
}

func cacheTestToken(t *testing.T, id string, bound bool) string {
	t.Helper()
	claims := map[string]any{
		"iss": "https://issuer.test", "aud": []string{"https://webhook.test"},
		"sub": "system:serviceaccount:test:hook", "jti": id,
		"nbf": time.Now().Add(-time.Minute).Unix(), "exp": time.Now().Add(time.Hour).Unix(),
		"kubernetes.io": map[string]any{
			"attestations": map[string]any{admissionReviewAPIGroupsClaimKey: []string{"apps"}},
		},
	}
	if bound {
		claims["kubernetes.io"].(map[string]any)["validatingwebhookconfiguration"] = map[string]any{
			"name": "hook", "uid": "hook-uid",
		}
	}
	payload, err := json.Marshal(claims)
	if err != nil {
		t.Fatalf("marshal cache test token: %v", err)
	}
	header := base64.RawURLEncoding.EncodeToString([]byte(`{"alg":"RS256","typ":"JWT"}`))
	return header + "." + base64.RawURLEncoding.EncodeToString(payload) + "." + base64.RawURLEncoding.EncodeToString([]byte("signature"))
}

func TestVerifiedTokenCacheSkipsRepeatedSignatureOnlyForValidClaims(t *testing.T) {
	// Real-signature tests cover trust; this KeySet seam counts crypto calls so
	// an accidentally broader cache cannot hide claim-policy failures.
	keys := &countingKeySet{}
	verifier := coreosoidc.NewVerifier("https://issuer.test", keys, &coreosoidc.Config{ClientID: "https://webhook.test"})
	cache := newTokenCache()
	valid := cacheTestToken(t, "valid", true)

	first, err := verifyTokenGroups(context.Background(), verifier, "https://issuer.test", valid, cache)
	if err != nil || len(first) != 1 || first[0] != "apps" {
		t.Fatalf("first valid token: groups=%v err=%v", first, err)
	}
	first[0] = "modified"
	second, err := verifyTokenGroups(context.Background(), verifier, "https://issuer.test", valid, cache)
	if err != nil || len(second) != 1 || second[0] != "apps" || keys.calls.Load() != 1 {
		t.Fatalf("cached valid token: groups=%v err=%v signatureChecks=%d", second, err, keys.calls.Load())
	}

	invalid := cacheTestToken(t, "unbound", false)
	for range 2 {
		if _, err := verifyTokenGroups(context.Background(), verifier, "https://issuer.test", invalid, cache); err == nil {
			t.Fatal("unbound ServiceAccount token was accepted or cached")
		}
	}
	if got := keys.calls.Load(); got != 3 {
		t.Fatalf("invalid token signature checks = %d, want two uncached checks", got-1)
	}

	stale := cacheTestToken(t, "stale", true)
	cache.Add(sha256.Sum256([]byte(stale)), verifiedToken{
		groups: []string{"*"}, expiry: time.Now().Add(-time.Second),
	}, time.Minute)
	groups, err := verifyTokenGroups(context.Background(), verifier, "https://issuer.test", stale, cache)
	if err != nil || len(groups) != 1 || groups[0] != "apps" || keys.calls.Load() != 4 {
		t.Fatalf("expired cached entry: groups=%v err=%v signatureChecks=%d", groups, err, keys.calls.Load())
	}
}
