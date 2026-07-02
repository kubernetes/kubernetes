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

package verify

import (
	"context"
	"encoding/json"
	"errors"
	"testing"
	"time"
)

// fakeKeySet is a stand-in KeySet that exercises the core contract logic without
// any real crypto. When err is set it simulates a signature failure; otherwise
// it treats the raw token string as the already-verified JSON payload. This lets
// every contract check run fully offline with no third-party dependency.
type fakeKeySet struct {
	err error
}

func (f fakeKeySet) VerifySignature(_ context.Context, rawToken string) ([]byte, error) {
	if f.err != nil {
		return nil, f.err
	}
	return []byte(rawToken), nil
}

// fixedClock returns a now func pinned to t.
func fixedClock(t time.Time) func() time.Time {
	return func() time.Time { return t }
}

const (
	testAudience = "webhook.example.com"
	testGroup    = "apps"
	testIssuer   = "https://issuer.example.com"
)

// tokenBuilder assembles a token payload for tests. Because the fake KeySet
// passes the raw token through as the payload, "signing" is just JSON marshaling.
type tokenBuilder struct {
	aud    audience
	exp    *int64
	nbf    *int64
	k8s    *kubernetesClaims
	sub    string
	iss    string
	rawaud json.RawMessage // when set, overrides aud to test the single-string form
}

func baseToken() *tokenBuilder {
	exp := time.Now().Add(5 * time.Minute).Unix()
	return &tokenBuilder{
		aud: audience{testAudience},
		exp: &exp,
		sub: "system:serviceaccount:kube-system:webhook-auth",
		iss: "https://issuer.example.com",
		k8s: &kubernetesClaims{
			ValidatingWebhookConfiguration: &ObjectRef{Name: "vwc", UID: "vwc-uid"},
			AttestationClaims: map[string][]string{
				AllowedAPIGroupClaimKey: {testGroup},
			},
		},
	}
}

func (b *tokenBuilder) build(t *testing.T) string {
	t.Helper()
	// Marshal into a generic map so we can control exact JSON keys (e.g. the
	// "kubernetes.io" key and numeric-date encodings).
	m := map[string]interface{}{}
	if b.rawaud != nil {
		m["aud"] = b.rawaud
	} else if b.aud != nil {
		m["aud"] = []string(b.aud)
	}
	if b.exp != nil {
		m["exp"] = *b.exp
	}
	if b.nbf != nil {
		m["nbf"] = *b.nbf
	}
	if b.sub != "" {
		m["sub"] = b.sub
	}
	if b.iss != "" {
		m["iss"] = b.iss
	}
	if b.k8s != nil {
		m["kubernetes.io"] = b.k8s
	}
	raw, err := json.Marshal(m)
	if err != nil {
		t.Fatalf("marshal token: %v", err)
	}
	return string(raw)
}

func newTestVerifier(t *testing.T, ks KeySet, opts ...Option) *Verifier {
	t.Helper()
	v, err := NewVerifier(ks, testIssuer, []string{testAudience}, opts...)
	if err != nil {
		t.Fatalf("NewVerifier: %v", err)
	}
	return v
}

func TestNewVerifier_Validation(t *testing.T) {
	tests := []struct {
		name      string
		keySet    KeySet
		issuer    string
		audiences []string
		wantErr   bool
	}{
		{
			name:      "nil KeySet -> error",
			keySet:    nil,
			issuer:    testIssuer,
			audiences: []string{testAudience},
			wantErr:   true,
		},
		{
			name:      "empty issuer -> error",
			keySet:    fakeKeySet{},
			issuer:    "",
			audiences: []string{testAudience},
			wantErr:   true,
		},
		{
			name:      "empty audiences -> error",
			keySet:    fakeKeySet{},
			issuer:    testIssuer,
			audiences: nil,
			wantErr:   true,
		},
		{
			name:      "valid KeySet, issuer and audience -> ok",
			keySet:    fakeKeySet{},
			issuer:    testIssuer,
			audiences: []string{testAudience},
			wantErr:   false,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := NewVerifier(tc.keySet, tc.issuer, tc.audiences)
			if tc.wantErr && err == nil {
				t.Fatal("expected error, got nil")
			}
			if !tc.wantErr && err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}
}

// TestVerify exercises the core Verify contract as a table of happy and sad
// paths. Each case starts from baseToken(), optionally mutates it, and asserts
// either a successful Result (happy path) or, for a sad path, that the failure
// (a) satisfies the single generic ErrVerificationFailed and (b) reports the
// expected reason string via Reason so we still prove the right check fired.
func TestVerify(t *testing.T) {
	// leewayExpiry is the fixed exp instant used by the clock-skew cases so the
	// token boundary is deterministic relative to the injected clock.
	leewayExpiry := time.Unix(1_000_000, 0)
	unixPtr := func(t time.Time) *int64 { u := t.Unix(); return &u }

	tests := []struct {
		name string
		// keySet overrides the signature backend; nil defaults to a passing fakeKeySet.
		keySet KeySet
		// mutate customizes the base token before it is built; nil leaves it as-is.
		mutate func(b *tokenBuilder)
		// rawToken, when non-empty, is verified verbatim instead of a built token.
		rawToken string
		// opts are passed to NewVerifier (e.g. clock, leeway).
		opts []Option
		// group is the reviewAPIGroup argument; empty defaults to testGroup.
		group string
		// wantReason is the expected log reason; empty means the happy path is expected.
		wantReason string
		// check runs extra assertions on the Result of a happy-path case.
		check func(t *testing.T, res *Result)
	}{
		{
			name: "validating webhook config, exact group -> accepted",
			check: func(t *testing.T, res *Result) {
				if res.BoundObjectKind != KindValidatingWebhookConfiguration {
					t.Errorf("BoundObjectKind = %q, want %q", res.BoundObjectKind, KindValidatingWebhookConfiguration)
				}
				if res.BoundObjectName != "vwc" || res.BoundObjectUID != "vwc-uid" {
					t.Errorf("bound object = %s/%s, want vwc/vwc-uid", res.BoundObjectName, res.BoundObjectUID)
				}
				if res.AllowedAPIGroup != testGroup {
					t.Errorf("AllowedAPIGroup = %q, want %q", res.AllowedAPIGroup, testGroup)
				}
				if res.Subject != "system:serviceaccount:kube-system:webhook-auth" {
					t.Errorf("Subject = %q", res.Subject)
				}
				if len(res.Audience) != 1 || res.Audience[0] != testAudience {
					t.Errorf("Audience = %v", res.Audience)
				}
			},
		},
		{
			name: "mutating webhook config, exact group -> accepted",
			mutate: func(b *tokenBuilder) {
				b.k8s.ValidatingWebhookConfiguration = nil
				b.k8s.MutatingWebhookConfiguration = &ObjectRef{Name: "mwc", UID: "mwc-uid"}
			},
			check: func(t *testing.T, res *Result) {
				if res.BoundObjectKind != KindMutatingWebhookConfiguration {
					t.Errorf("BoundObjectKind = %q, want %q", res.BoundObjectKind, KindMutatingWebhookConfiguration)
				}
				if res.BoundObjectName != "mwc" {
					t.Errorf("BoundObjectName = %q, want mwc", res.BoundObjectName)
				}
			},
		},
		{
			name: "exact allowedAPIGroup equals review group -> accepted",
			// Base token already carries testGroup; confirms the exact-match branch.
		},
		{
			name: "wildcard allowedAPIGroup -> matches any review group -> accepted",
			mutate: func(b *tokenBuilder) {
				b.k8s.AttestationClaims = map[string][]string{AllowedAPIGroupClaimKey: {WildcardAPIGroup}}
			},
			group: "any.group.example.com",
			check: func(t *testing.T, res *Result) {
				if res.AllowedAPIGroup != WildcardAPIGroup {
					t.Errorf("AllowedAPIGroup = %q, want %q", res.AllowedAPIGroup, WildcardAPIGroup)
				}
			},
		},
		{
			name: "audience as a bare RFC 7519 string -> accepted",
			// RFC 7519 permits "aud" as a single string rather than an array.
			mutate: func(b *tokenBuilder) {
				b.rawaud = json.RawMessage(`"` + testAudience + `"`)
			},
		},
		{
			name: "nbf in the past -> accepted",
			mutate: func(b *tokenBuilder) {
				past := time.Now().Add(-1 * time.Hour).Unix()
				b.nbf = &past
			},
		},
		{
			name: "expired within leeway window -> tolerated -> accepted",
			// exp is 30s in the past but a 60s leeway absorbs the skew.
			mutate: func(b *tokenBuilder) { b.exp = unixPtr(leewayExpiry) },
			opts: []Option{
				WithClock(fixedClock(leewayExpiry.Add(30 * time.Second))),
				WithLeeway(60 * time.Second),
			},
		},
		{
			name: "both bound objects set -> reasonBothBoundObjects",
			mutate: func(b *tokenBuilder) {
				b.k8s.MutatingWebhookConfiguration = &ObjectRef{Name: "mwc", UID: "mwc-uid"}
			},
			wantReason: reasonBothBoundObjects,
		},
		{
			name: "neither bound object set -> reasonNoBoundObject",
			mutate: func(b *tokenBuilder) {
				b.k8s.ValidatingWebhookConfiguration = nil
			},
			wantReason: reasonNoBoundObject,
		},
		{
			name: "missing kubernetes.io claims -> reasonNoBoundObject",
			mutate: func(b *tokenBuilder) {
				b.k8s = nil
			},
			wantReason: reasonNoBoundObject,
		},
		{
			name: "audience does not include expected -> reasonAudienceMismatch",
			mutate: func(b *tokenBuilder) {
				b.aud = audience{"someone.else.example.com"}
			},
			wantReason: reasonAudienceMismatch,
		},
		{
			name: "issuer differs from expected -> reasonIssuerMismatch",
			mutate: func(b *tokenBuilder) {
				b.iss = "https://attacker.example.com"
			},
			wantReason: reasonIssuerMismatch,
		},
		{
			name: "issuer absent -> reasonIssuerMismatch",
			mutate: func(b *tokenBuilder) {
				b.iss = ""
			},
			wantReason: reasonIssuerMismatch,
		},
		{
			name: "expiry claim absent -> reasonMissingExpiry",
			mutate: func(b *tokenBuilder) {
				b.exp = nil
			},
			wantReason: reasonMissingExpiry,
		},
		{
			name: "allowedAPIGroup claim absent -> reasonMissingAllowedAPIGroup",
			mutate: func(b *tokenBuilder) {
				b.k8s.AttestationClaims = map[string][]string{}
			},
			wantReason: reasonMissingAllowedAPIGroup,
		},
		{
			name: "bare allowedAPIGroup key -> treated as missing -> reasonMissingAllowedAPIGroup",
			// A spec-violating issuer emitting the bare "allowedAPIGroup" key (not
			// the namespaced form) must be treated as missing, never accepted.
			mutate: func(b *tokenBuilder) {
				b.k8s.AttestationClaims = map[string][]string{"allowedAPIGroup": {testGroup}}
			},
			wantReason: reasonMissingAllowedAPIGroup,
		},
		{
			name: "allowedAPIGroup with more than one value -> reasonMissingAllowedAPIGroup",
			mutate: func(b *tokenBuilder) {
				b.k8s.AttestationClaims = map[string][]string{AllowedAPIGroupClaimKey: {testGroup, "extensions"}}
			},
			wantReason: reasonMissingAllowedAPIGroup,
		},
		{
			name:       "review group not authorized by claim -> reasonAPIGroupNotAuthorized",
			group:      "batch",
			wantReason: reasonAPIGroupNotAuthorized,
		},
		{
			name: "token past exp -> reasonExpired",
			mutate: func(b *tokenBuilder) {
				past := time.Now().Add(-1 * time.Hour).Unix()
				b.exp = &past
			},
			wantReason: reasonExpired,
		},
		{
			name: "expired beyond leeway window -> reasonExpired",
			// exp is 2m in the past; a 60s leeway is not enough to tolerate it.
			mutate: func(b *tokenBuilder) { b.exp = unixPtr(leewayExpiry) },
			opts: []Option{
				WithClock(fixedClock(leewayExpiry.Add(2 * time.Minute))),
				WithLeeway(60 * time.Second),
			},
			wantReason: reasonExpired,
		},
		{
			name: "leeway request above the cap is clamped -> reasonExpired",
			// exp is 6m in the past. An over-large 10m leeway is clamped to the 5m
			// maximum, which is not enough to tolerate the skew, so the token is
			// still rejected. This proves WithLeeway cannot neuter expiry.
			mutate: func(b *tokenBuilder) { b.exp = unixPtr(leewayExpiry) },
			opts: []Option{
				WithClock(fixedClock(leewayExpiry.Add(6 * time.Minute))),
				WithLeeway(10 * time.Minute),
			},
			wantReason: reasonExpired,
		},
		{
			name: "token before nbf -> reasonNotYetValid",
			mutate: func(b *tokenBuilder) {
				future := time.Now().Add(1 * time.Hour).Unix()
				b.nbf = &future
			},
			wantReason: reasonNotYetValid,
		},
		{
			name:       "signature does not verify -> reasonInvalidSignature",
			keySet:     fakeKeySet{err: errors.New("no matching key")},
			wantReason: reasonInvalidSignature,
		},
		{
			name:       "payload is not JSON -> reasonMalformedToken",
			rawToken:   "this is not json",
			wantReason: reasonMalformedToken,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ks := tc.keySet
			if ks == nil {
				ks = fakeKeySet{}
			}
			v := newTestVerifier(t, ks, tc.opts...)

			tok := tc.rawToken
			if tok == "" {
				b := baseToken()
				if tc.mutate != nil {
					tc.mutate(b)
				}
				tok = b.build(t)
			}

			group := tc.group
			if group == "" {
				group = testGroup
			}

			res, err := v.Verify(context.Background(), tok, group)
			if tc.wantReason != "" {
				assertFailure(t, err, tc.wantReason)
				return
			}
			if err != nil {
				t.Fatalf("Verify: unexpected error: %v", err)
			}
			if tc.check != nil {
				tc.check(t, res)
			}
		})
	}
}

func TestVerify_ErrorsAreGenericAndAntiEnumeration(t *testing.T) {
	v := newTestVerifier(t, fakeKeySet{})
	// A token bound to a specific webhook/uid, mismatched group.
	tok := baseToken().build(t)
	_, err := v.Verify(context.Background(), tok, "batch")
	if err == nil {
		t.Fatal("expected error")
	}
	// Every failure must satisfy the generic sentinel...
	if !errors.Is(err, ErrVerificationFailed) {
		t.Errorf("error %v does not satisfy ErrVerificationFailed", err)
	}
	// ...and must not leak the webhook name, uid, or the rejected group value.
	msg := err.Error()
	for _, leak := range []string{"vwc", "vwc-uid", "batch", testGroup} {
		if containsSubstring(msg, leak) {
			t.Errorf("error message %q leaks %q", msg, leak)
		}
	}
}

// assertFailure checks that err is a verification failure: it satisfies the
// single generic ErrVerificationFailed and its Reason contains wantReason, so
// we prove the right check fired without asserting a distinct exported sentinel.
func assertFailure(t *testing.T, err error, wantReason string) {
	t.Helper()
	if err == nil {
		t.Fatalf("expected failure with reason %q, got nil", wantReason)
	}
	if !errors.Is(err, ErrVerificationFailed) {
		t.Errorf("error %v does not satisfy generic ErrVerificationFailed", err)
	}
	if got := Reason(err); !containsSubstring(got, wantReason) {
		t.Errorf("reason = %q, want it to contain %q", got, wantReason)
	}
}

func containsSubstring(haystack, needle string) bool {
	if needle == "" {
		return false
	}
	for i := 0; i+len(needle) <= len(haystack); i++ {
		if haystack[i:i+len(needle)] == needle {
			return true
		}
	}
	return false
}
