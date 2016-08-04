/*
Copyright 2016 The Kubernetes Authors.

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
	"encoding/base64"
	"errors"
	"fmt"
	"io/ioutil"
	"net/http"
	"os"
	"path"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/coreos/go-oidc/jose"
	"github.com/coreos/go-oidc/key"
	"github.com/coreos/go-oidc/oauth2"

	"k8s.io/kubernetes/pkg/util/diff"
	"k8s.io/kubernetes/pkg/util/wait"
	oidctesting "k8s.io/kubernetes/plugin/pkg/auth/authenticator/token/oidc/testing"
)

func TestNewOIDCAuthProvider(t *testing.T) {
	tempDir, err := ioutil.TempDir(os.TempDir(), "oidc_test")
	if err != nil {
		t.Fatalf("Cannot make temp dir %v", err)
	}
	cert := path.Join(tempDir, "oidc-cert")
	key := path.Join(tempDir, "oidc-key")

	defer os.Remove(cert)
	defer os.Remove(key)
	defer os.Remove(tempDir)

	oidctesting.GenerateSelfSignedCert(t, "127.0.0.1", cert, key)
	op := oidctesting.NewOIDCProvider(t, "")
	srv, err := op.ServeTLSWithKeyPair(cert, key)
	if err != nil {
		t.Fatalf("Cannot start server %v", err)
	}
	defer srv.Close()

	certData, err := ioutil.ReadFile(cert)
	if err != nil {
		t.Fatalf("Could not read cert bytes %v", err)
	}

	jwt, err := jose.NewSignedJWT(jose.Claims(map[string]interface{}{
		"test": "jwt",
	}), op.PrivKey.Signer())
	if err != nil {
		t.Fatalf("Could not create signed JWT %v", err)
	}

	tests := []struct {
		cfg map[string]string

		wantErr            bool
		wantInitialIDToken jose.JWT
	}{
		{
			// A Valid configuration
			cfg: map[string]string{
				cfgIssuerUrl:            srv.URL,
				cfgCertificateAuthority: cert,
				cfgClientID:             "client-id",
				cfgClientSecret:         "client-secret",
			},
		},
		{
			// A Valid configuration with an Initial JWT
			cfg: map[string]string{
				cfgIssuerUrl:            srv.URL,
				cfgCertificateAuthority: cert,
				cfgClientID:             "client-id",
				cfgClientSecret:         "client-secret",
				cfgIDToken:              jwt.Encode(),
			},
			wantInitialIDToken: *jwt,
		},
		{
			// Valid config, but using cfgCertificateAuthorityData
			cfg: map[string]string{
				cfgIssuerUrl:                srv.URL,
				cfgCertificateAuthorityData: base64.StdEncoding.EncodeToString(certData),
				cfgClientID:                 "client-id",
				cfgClientSecret:             "client-secret",
			},
		},
		{
			// Missing client id
			cfg: map[string]string{
				cfgIssuerUrl:            srv.URL,
				cfgCertificateAuthority: cert,
				cfgClientSecret:         "client-secret",
			},
			wantErr: true,
		},
		{
			// Missing client secret
			cfg: map[string]string{
				cfgIssuerUrl:            srv.URL,
				cfgCertificateAuthority: cert,
				cfgClientID:             "client-id",
			},
			wantErr: true,
		},
		{
			// Missing issuer url.
			cfg: map[string]string{
				cfgCertificateAuthority: cert,
				cfgClientID:             "client-id",
				cfgClientSecret:         "secret",
			},
			wantErr: true,
		},
		{
			// No TLS config
			cfg: map[string]string{
				cfgIssuerUrl:    srv.URL,
				cfgClientID:     "client-id",
				cfgClientSecret: "secret",
			},
			wantErr: true,
		},
	}

	for i, tt := range tests {
		ap, err := newOIDCAuthProvider("cluster.example.com", tt.cfg, nil)
		if tt.wantErr {
			if err == nil {
				t.Errorf("case %d: want non-nil err", i)
			}
			continue
		}

		if err != nil {
			t.Errorf("case %d: unexpected error on newOIDCAuthProvider: %v", i, err)
			continue
		}

		oidcAP, ok := ap.(*oidcAuthProvider)
		if !ok {
			t.Errorf("case %d: expected ap to be an oidcAuthProvider", i)
			continue
		}

		if diff := compareJWTs(tt.wantInitialIDToken, oidcAP.initialIDToken); diff != "" {
			t.Errorf("case %d: compareJWTs(tt.wantInitialIDToken, oidcAP.initialIDToken)=%v", i, diff)
		}
	}
}

func TestWrapTranport(t *testing.T) {
	oldBackoff := backoff
	defer func() {
		backoff = oldBackoff
	}()
	backoff = wait.Backoff{
		Duration: 1 * time.Nanosecond,
		Steps:    3,
	}

	privKey, err := key.GeneratePrivateKey()
	if err != nil {
		t.Fatalf("can't generate private key: %v", err)
	}

	makeToken := func(s string, exp time.Time, count int) *jose.JWT {
		jwt, err := jose.NewSignedJWT(jose.Claims(map[string]interface{}{
			"test":  s,
			"exp":   exp.UTC().Unix(),
			"count": count,
		}), privKey.Signer())
		if err != nil {
			t.Fatalf("Could not create signed JWT %v", err)
		}
		return jwt
	}

	goodToken := makeToken("good", time.Now().Add(time.Hour), 0)
	goodToken2 := makeToken("good", time.Now().Add(time.Hour), 1)
	expiredToken := makeToken("good", time.Now().Add(-time.Hour), 0)

	str := func(s string) *string {
		return &s
	}
	tests := []struct {
		cfgIDToken      *jose.JWT
		cfgRefreshToken *string

		expectRequests []testRoundTrip

		expectRefreshes []testRefresh

		expectPersists []testPersist

		wantStatus int
		wantErr    bool
	}{
		{
			// Initial JWT is set, it is good, it is set as bearer.
			cfgIDToken: goodToken,

			expectRequests: []testRoundTrip{
				{
					expectBearerToken: goodToken.Encode(),
					returnHTTPStatus:  200,
				},
			},

			wantStatus: 200,
		},
		{
			// Initial JWT is set, but it's expired, so it gets refreshed.
			cfgIDToken:      expiredToken,
			cfgRefreshToken: str("rt1"),

			expectRefreshes: []testRefresh{
				{
					expectRefreshToken: "rt1",
					returnTokens: oauth2.TokenResponse{
						IDToken: goodToken.Encode(),
					},
				},
			},

			expectRequests: []testRoundTrip{
				{
					expectBearerToken: goodToken.Encode(),
					returnHTTPStatus:  200,
				},
			},

			expectPersists: []testPersist{
				{
					cfg: map[string]string{
						cfgIDToken:      goodToken.Encode(),
						cfgRefreshToken: "rt1",
					},
				},
			},

			wantStatus: 200,
		},
		{
			// Initial JWT is set, but it's expired, so it gets refreshed - this
			// time the refresh token itself is also refreshed
			cfgIDToken:      expiredToken,
			cfgRefreshToken: str("rt1"),

			expectRefreshes: []testRefresh{
				{
					expectRefreshToken: "rt1",
					returnTokens: oauth2.TokenResponse{
						IDToken:      goodToken.Encode(),
						RefreshToken: "rt2",
					},
				},
			},

			expectRequests: []testRoundTrip{
				{
					expectBearerToken: goodToken.Encode(),
					returnHTTPStatus:  200,
				},
			},

			expectPersists: []testPersist{
				{
					cfg: map[string]string{
						cfgIDToken:      goodToken.Encode(),
						cfgRefreshToken: "rt2",
					},
				},
			},

			wantStatus: 200,
		},
		{
			// Initial JWT is not set, so it gets refreshed.
			cfgRefreshToken: str("rt1"),

			expectRefreshes: []testRefresh{
				{
					expectRefreshToken: "rt1",
					returnTokens: oauth2.TokenResponse{
						IDToken: goodToken.Encode(),
					},
				},
			},

			expectRequests: []testRoundTrip{
				{
					expectBearerToken: goodToken.Encode(),
					returnHTTPStatus:  200,
				},
			},

			expectPersists: []testPersist{
				{
					cfg: map[string]string{
						cfgIDToken:      goodToken.Encode(),
						cfgRefreshToken: "rt1",
					},
				},
			},

			wantStatus: 200,
		},
		{
			// Expired token, but no refresh token.
			cfgIDToken: expiredToken,

			wantErr: true,
		},
		{
			// Initial JWT is not set, so it gets refreshed, but the server
			// rejects it when it is used, so it refreshes again, which
			// succeeds.
			cfgRefreshToken: str("rt1"),

			expectRefreshes: []testRefresh{
				{
					expectRefreshToken: "rt1",
					returnTokens: oauth2.TokenResponse{
						IDToken: goodToken.Encode(),
					},
				},
				{
					expectRefreshToken: "rt1",
					returnTokens: oauth2.TokenResponse{
						IDToken: goodToken2.Encode(),
					},
				},
			},

			expectRequests: []testRoundTrip{
				{
					expectBearerToken: goodToken.Encode(),
					returnHTTPStatus:  http.StatusUnauthorized,
				},
				{
					expectBearerToken: goodToken2.Encode(),
					returnHTTPStatus:  http.StatusOK,
				},
			},

			expectPersists: []testPersist{
				{
					cfg: map[string]string{
						cfgIDToken:      goodToken.Encode(),
						cfgRefreshToken: "rt1",
					},
				},
				{
					cfg: map[string]string{
						cfgIDToken:      goodToken2.Encode(),
						cfgRefreshToken: "rt1",
					},
				},
			},

			wantStatus: 200,
		},
		{
			// Initial JWT is but the server rejects it when it is used, so it
			// refreshes again, which succeeds.
			cfgRefreshToken: str("rt1"),
			cfgIDToken:      goodToken,

			expectRefreshes: []testRefresh{
				{
					expectRefreshToken: "rt1",
					returnTokens: oauth2.TokenResponse{
						IDToken: goodToken2.Encode(),
					},
				},
			},

			expectRequests: []testRoundTrip{
				{
					expectBearerToken: goodToken.Encode(),
					returnHTTPStatus:  http.StatusUnauthorized,
				},
				{
					expectBearerToken: goodToken2.Encode(),
					returnHTTPStatus:  http.StatusOK,
				},
			},

			expectPersists: []testPersist{
				{
					cfg: map[string]string{
						cfgIDToken:      goodToken2.Encode(),
						cfgRefreshToken: "rt1",
					},
				},
			},
			wantStatus: 200,
		},
	}

	for i, tt := range tests {
		client := &testOIDCClient{
			refreshes: tt.expectRefreshes,
		}

		persister := &testPersister{
			tt.expectPersists,
		}

		cfg := map[string]string{}
		if tt.cfgIDToken != nil {
			cfg[cfgIDToken] = tt.cfgIDToken.Encode()
		}

		if tt.cfgRefreshToken != nil {
			cfg[cfgRefreshToken] = *tt.cfgRefreshToken
		}

		ap := &oidcAuthProvider{
			refresher: &idTokenRefresher{
				client:    client,
				cfg:       cfg,
				persister: persister,
			},
		}

		if tt.cfgIDToken != nil {
			ap.initialIDToken = *tt.cfgIDToken
		}

		tstRT := &testRoundTripper{
			tt.expectRequests,
		}

		rt := ap.WrapTransport(tstRT)

		req, err := http.NewRequest("GET", "http://cluster.example.com", nil)
		if err != nil {
			t.Errorf("case %d: unexpected error making request: %v", i, err)
		}

		res, err := rt.RoundTrip(req)
		if tt.wantErr {
			if err == nil {
				t.Errorf("case %d: Expected non-nil error", i)
			}
		} else if err != nil {
			t.Errorf("case %d: unexpected error making round trip: %v", i, err)

		} else {
			if res.StatusCode != tt.wantStatus {
				t.Errorf("case %d: want=%d, got=%d", i, tt.wantStatus, res.StatusCode)
			}
		}

		if err = client.verify(); err != nil {
			t.Errorf("case %d: %v", i, err)
		}

		if err = persister.verify(); err != nil {
			t.Errorf("case %d: %v", i, err)
		}

		if err = tstRT.verify(); err != nil {
			t.Errorf("case %d: %v", i, err)
			continue
		}

	}
}

type testRoundTrip struct {
	expectBearerToken string
	returnHTTPStatus  int
}

type testRoundTripper struct {
	trips []testRoundTrip
}

func (t *testRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	if len(t.trips) == 0 {
		return nil, errors.New("unexpected RoundTrip call")
	}

	var trip testRoundTrip
	trip, t.trips = t.trips[0], t.trips[1:]

	var bt string
	var parts []string
	auth := strings.TrimSpace(req.Header.Get("Authorization"))
	if auth == "" {
		goto Compare
	}

	parts = strings.Split(auth, " ")
	if len(parts) < 2 || strings.ToLower(parts[0]) != "bearer" {
		goto Compare
	}

	bt = parts[1]

Compare:
	if trip.expectBearerToken != bt {
		return nil, fmt.Errorf("want bearerToken=%v, got=%v", trip.expectBearerToken, bt)
	}
	return &http.Response{
		StatusCode: trip.returnHTTPStatus,
	}, nil
}

func (t *testRoundTripper) verify() error {
	if l := len(t.trips); l > 0 {
		return fmt.Errorf("%d uncalled round trips", l)
	}
	return nil
}

type testPersist struct {
	cfg       map[string]string
	returnErr error
}

type testPersister struct {
	persists []testPersist
}

func (t *testPersister) Persist(cfg map[string]string) error {
	if len(t.persists) == 0 {
		return errors.New("unexpected persist call")
	}

	var persist testPersist
	persist, t.persists = t.persists[0], t.persists[1:]

	if !reflect.DeepEqual(persist.cfg, cfg) {
		return fmt.Errorf("Unexpected cfg: %v", diff.ObjectDiff(persist.cfg, cfg))
	}

	return persist.returnErr
}

func (t *testPersister) verify() error {
	if l := len(t.persists); l > 0 {
		return fmt.Errorf("%d uncalled persists", l)
	}
	return nil
}

type testRefresh struct {
	expectRefreshToken string

	returnErr    error
	returnTokens oauth2.TokenResponse
}

type testOIDCClient struct {
	refreshes []testRefresh
}

func (o *testOIDCClient) refreshToken(rt string) (oauth2.TokenResponse, error) {
	if len(o.refreshes) == 0 {
		return oauth2.TokenResponse{}, errors.New("unexpected refresh request")
	}

	var refresh testRefresh
	refresh, o.refreshes = o.refreshes[0], o.refreshes[1:]

	if rt != refresh.expectRefreshToken {
		return oauth2.TokenResponse{}, fmt.Errorf("want rt=%v, got=%v",
			refresh.expectRefreshToken,
			rt)
	}

	if refresh.returnErr != nil {
		return oauth2.TokenResponse{}, refresh.returnErr
	}

	return refresh.returnTokens, nil
}

func (o *testOIDCClient) verifyJWT(jwt jose.JWT) error {
	claims, err := jwt.Claims()
	if err != nil {
		return err
	}
	claim, _, _ := claims.StringClaim("test")
	if claim != "good" {
		return errors.New("bad token")
	}
	return nil
}

func (t *testOIDCClient) verify() error {
	if l := len(t.refreshes); l > 0 {
		return fmt.Errorf("%d uncalled refreshes", l)
	}
	return nil
}

func compareJWTs(a, b jose.JWT) string {
	if a.Encode() == b.Encode() {
		return ""
	}

	var aClaims, bClaims jose.Claims
	for _, j := range []struct {
		claims *jose.Claims
		jwt    jose.JWT
	}{
		{&aClaims, a},
		{&bClaims, b},
	} {
		var err error
		*j.claims, err = j.jwt.Claims()
		if err != nil {
			*j.claims = jose.Claims(map[string]interface{}{
				"msg": "bad claims",
				"err": err,
			})
		}
	}

	return diff.ObjectDiff(aClaims, bClaims)
}
