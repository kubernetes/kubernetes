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

package serviceaccount

import (
	"crypto/rand"
	"crypto/rsa"
	"strings"
	"testing"
	"time"

	"gopkg.in/go-jose/go-jose.v2/jwt"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/authentication/authenticator"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/metrics/testutil"
	"k8s.io/klog/v2/ktesting"
)

func TestLegacyTokensMetric(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	RegisterMetrics()
	legacyTokensTotal.Reset()
	defer legacyTokensTotal.Reset()

	privKey, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		t.Fatalf("generate RSA key: %v", err)
	}

	sa := v1.ServiceAccount{ObjectMeta: metav1.ObjectMeta{Name: "saname", Namespace: "ns", UID: "sauid"}}
	sec := v1.Secret{ObjectMeta: metav1.ObjectMeta{Name: "secname", Namespace: "ns"}}

	gen, err := JWTTokenGenerator(LegacyIssuer, privKey)
	if err != nil {
		t.Fatalf("generator: %v", err)
	}
	public, private := LegacyClaims(sa, sec)
	token, err := gen.GenerateToken(ctx, public, private)
	if err != nil {
		t.Fatalf("generate token: %v", err)
	}

	legacyValidator, err := NewLegacyValidator(false, nil, nil)
	if err != nil {
		t.Fatalf("validator: %v", err)
	}
	keysGetter, err := StaticPublicKeysGetter([]interface{}{&privKey.PublicKey})
	if err != nil {
		t.Fatalf("keys getter: %v", err)
	}
	authn := JWTTokenAuthenticator[legacyPrivateClaims](
		[]string{LegacyIssuer},
		keysGetter,
		authenticator.Audiences{"api"},
		legacyValidator,
	)

	if _, ok, err := authn.AuthenticateToken(ctx, token); err != nil || !ok {
		t.Fatalf("AuthenticateToken failed: ok=%v err=%v", ok, err)
	}

	want := `
		# HELP serviceaccount_legacy_tokens_total [BETA] Cumulative legacy service account tokens used
		# TYPE serviceaccount_legacy_tokens_total counter
		serviceaccount_legacy_tokens_total 1
	`
	if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(want), "serviceaccount_legacy_tokens_total"); err != nil {
		t.Fatal(err)
	}
}

func TestWarnAfterTokensMetric(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	RegisterMetrics()

	nowUnix := int64(1514764800)
	warnAfterUnix := nowUnix + 3600

	tests := []struct {
		name       string
		nowOffset  int64
		metricName string
		wantMetric string
	}{
		{
			name:       "stale when now is after warnAfter",
			nowOffset:  +1,
			metricName: "serviceaccount_stale_tokens_total",
			wantMetric: `
				# HELP serviceaccount_stale_tokens_total [BETA] Cumulative stale projected service account tokens used
				# TYPE serviceaccount_stale_tokens_total counter
				serviceaccount_stale_tokens_total 1
			`,
		},
		{
			name:       "valid when now is before warnAfter",
			nowOffset:  -1,
			metricName: "serviceaccount_valid_tokens_total",
			wantMetric: `
				# HELP serviceaccount_valid_tokens_total [BETA] Cumulative valid projected service account tokens used
				# TYPE serviceaccount_valid_tokens_total counter
				serviceaccount_valid_tokens_total 1
			`,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			staleTokensTotal.Reset()
			validTokensTotal.Reset()

			originalNow := now
			now = func() time.Time { return time.Unix(warnAfterUnix+tc.nowOffset, 0) }
			defer func() { now = originalNow }()

			sa := &v1.ServiceAccount{
				ObjectMeta: metav1.ObjectMeta{Name: "saname", Namespace: "ns", UID: "sauid"},
			}
			v := &validator{getter: fakeGetter{serviceAccount: sa}}

			expiry := jwt.NumericDate(nowUnix + 100000)
			warn := jwt.NewNumericDate(time.Unix(warnAfterUnix, 0))
			private := &privateClaims{Kubernetes: kubernetes{
				Svcacct:   ref{Name: "saname", UID: "sauid"},
				Namespace: "ns",
				WarnAfter: warn,
			}}
			public := &jwt.Claims{Expiry: &expiry}

			if _, err := v.Validate(ctx, "", public, private); err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(tc.wantMetric), tc.metricName); err != nil {
				t.Fatal(err)
			}
		})
	}
}
