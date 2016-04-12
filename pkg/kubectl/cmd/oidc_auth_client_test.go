package cmd

import (
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"

	"github.com/coreos/go-oidc/jose"
	"github.com/coreos/go-oidc/key"

	"k8s.io/kubernetes/plugin/pkg/auth/authenticator/token/oidc"
	oidctesting "k8s.io/kubernetes/plugin/pkg/auth/authenticator/token/oidc/testing"
)

var (
	signingKey *key.PrivateKey
	signer     jose.Signer
)

func setup(t *testing.T) {
	var err error
	signingKey, err = key.GeneratePrivateKey()
	if err != nil {
		t.Fatalf("Could not generate private key: %v", err)
	}
	signer = signingKey.Signer()
}

func TestExchangeRefreshToken(t *testing.T) {
	setup(t)

	claims := map[string]interface{}{
		"claim": "claim",
	}
	testToken, err := jose.NewSignedJWT(claims, signer)
	if err != nil {
		t.Fatalf("Could not make test JWT: %v", err)
	}

	tests := []struct {
		// the request
		rt string

		// set up the handler
		expectRT string

		// the response from the FakeClient inside the handler

		returnErr     error
		returnIDToken jose.JWT

		// override the handler, and send different response back
		responseBody   string
		responseStatus int

		wantIDToken string
		wantErr     bool
	}{
		{
			// The Happy Path
			rt:       "good_rt",
			expectRT: "good_rt",

			returnIDToken: *testToken,

			wantIDToken: testToken.Encode(),
		},
		{
			// Unexpected refresh token
			rt:       "bad_rt",
			expectRT: "good_rt",

			wantErr: true,
		},
		{
			// Server Error
			rt: "good_rt",

			responseStatus: 500,

			wantErr: true,
		},
		{
			// Missing id_token in response.
			rt: "good_rt",

			responseStatus: 200,
			responseBody:   "",

			wantErr: true,
		},
	}

	for i, tt := range tests {
		client := &oidctesting.FakeClient{
			Err:                tt.returnErr,
			IDToken:            tt.returnIDToken,
			ExpectRefreshToken: tt.expectRT,
		}
		h := oidc.NewOIDCHTTPHandler(client)
		srv := httptest.NewUnstartedServer(http.HandlerFunc(
			func(w http.ResponseWriter, req *http.Request) {
				if tt.responseStatus != 0 {
					w.WriteHeader(tt.responseStatus)
					_, err := w.Write([]byte(tt.responseBody))
					if err != nil {
						t.Errorf("case %d: unexpected err writing response body: %v", err)
					}
					return
				}
				h.ServeHTTP(w, req)
			}))
		srv.Start()
		defer srv.Close()

		cli := OIDCAuthClient{
			server:     *mustParseURL(t, srv.URL),
			httpClient: http.DefaultClient,
		}

		idToken, err := cli.ExchangeRefreshToken(tt.rt)
		if tt.wantErr {
			if err == nil {
				t.Errorf("case %d: expecting non-nil error.", i)
			}
			continue
		}

		if err != nil {
			t.Errorf("case %d: unexpected error exchanging refresh token: %v", i, err)
		}

		if idToken != tt.wantIDToken {
			t.Errorf("case %d: want %v, got %v", tt.wantIDToken, idToken)
		}
	}
}

func mustParseURL(t *testing.T, s string) *url.URL {
	u, err := url.Parse(s)
	if err != nil {
		t.Fatalf("Failed to parse url: %v", err)
	}
	return u
}
