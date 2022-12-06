/*
Copyright 2019 The Kubernetes Authors.

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

package serviceaccount_test

import (
	"crypto/ecdsa"
	"crypto/rsa"
	"crypto/x509"
	"encoding/json"
	"math/big"
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"

	restful "github.com/emicklei/go-restful/v3"
	"github.com/google/go-cmp/cmp"
	jose "gopkg.in/square/go-jose.v2"

	"k8s.io/kubernetes/pkg/routes"
	"k8s.io/kubernetes/pkg/serviceaccount"
)

const (
	exampleIssuer = "https://issuer.example.com"
)

func setupServer(t *testing.T, iss string, keys []interface{}) (*httptest.Server, string) {
	t.Helper()

	c := restful.NewContainer()
	s := httptest.NewServer(c)

	// JWKS needs to be https, so swap that for the test
	jwksURI, err := url.Parse(s.URL)
	if err != nil {
		t.Fatal(err)
	}
	jwksURI.Scheme = "https"
	jwksURI.Path = serviceaccount.JWKSPath

	md, err := serviceaccount.NewOpenIDMetadata(
		iss, jwksURI.String(), "", keys)
	if err != nil {
		t.Fatal(err)
	}

	srv := routes.NewOpenIDMetadataServer(md.ConfigJSON, md.PublicKeysetJSON)
	srv.Install(c)

	return s, jwksURI.String()
}

var defaultKeys = []interface{}{getPublicKey(rsaPublicKey), getPublicKey(ecdsaPublicKey)}

// Configuration is an OIDC configuration, including most but not all required fields.
// https://openid.net/specs/openid-connect-discovery-1_0.html#ProviderMetadata
type Configuration struct {
	Issuer        string   `json:"issuer"`
	JWKSURI       string   `json:"jwks_uri"`
	ResponseTypes []string `json:"response_types_supported"`
	SigningAlgs   []string `json:"id_token_signing_alg_values_supported"`
	SubjectTypes  []string `json:"subject_types_supported"`
}

func TestServeConfiguration(t *testing.T) {
	s, jwksURI := setupServer(t, exampleIssuer, defaultKeys)
	defer s.Close()

	want := Configuration{
		Issuer:        exampleIssuer,
		JWKSURI:       jwksURI,
		ResponseTypes: []string{"id_token"},
		SubjectTypes:  []string{"public"},
		SigningAlgs:   []string{"ES256", "RS256"},
	}

	reqURL := s.URL + "/.well-known/openid-configuration"

	resp, err := http.Get(reqURL)
	if err != nil {
		t.Fatalf("Get(%s) = %v, %v want: <response>, <nil>", reqURL, resp, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Errorf("Get(%s) = %v, _ want: %v, _", reqURL, resp.StatusCode, http.StatusOK)
	}

	if got, want := resp.Header.Get("Content-Type"), "application/json"; got != want {
		t.Errorf("Get(%s) Content-Type = %q, _ want: %q, _", reqURL, got, want)
	}
	if got, want := resp.Header.Get("Cache-Control"), "public, max-age=3600"; got != want {
		t.Errorf("Get(%s) Cache-Control = %q, _ want: %q, _", reqURL, got, want)
	}

	var got Configuration
	if err := json.NewDecoder(resp.Body).Decode(&got); err != nil {
		t.Fatalf("Decode(_) = %v, want: <nil>", err)
	}

	if !cmp.Equal(want, got) {
		t.Errorf("unexpected diff in received configuration (-want, +got):%s",
			cmp.Diff(want, got))
	}
}

func TestServeKeys(t *testing.T) {
	wantPubRSA := getPublicKey(rsaPublicKey).(*rsa.PublicKey)
	wantPubECDSA := getPublicKey(ecdsaPublicKey).(*ecdsa.PublicKey)
	var serveKeysTests = []struct {
		Name     string
		Keys     []interface{}
		WantKeys []jose.JSONWebKey
	}{
		{
			Name: "configured public keys",
			Keys: []interface{}{
				getPublicKey(rsaPublicKey),
				getPublicKey(ecdsaPublicKey),
			},
			WantKeys: []jose.JSONWebKey{
				{
					Algorithm:    "RS256",
					Key:          wantPubRSA,
					KeyID:        rsaKeyID,
					Use:          "sig",
					Certificates: []*x509.Certificate{},
				},
				{
					Algorithm:    "ES256",
					Key:          wantPubECDSA,
					KeyID:        ecdsaKeyID,
					Use:          "sig",
					Certificates: []*x509.Certificate{},
				},
			},
		},
		{
			Name: "only publishes public keys",
			Keys: []interface{}{
				getPrivateKey(rsaPrivateKey),
				getPrivateKey(ecdsaPrivateKey),
			},
			WantKeys: []jose.JSONWebKey{
				{
					Algorithm:    "RS256",
					Key:          wantPubRSA,
					KeyID:        rsaKeyID,
					Use:          "sig",
					Certificates: []*x509.Certificate{},
				},
				{
					Algorithm:    "ES256",
					Key:          wantPubECDSA,
					KeyID:        ecdsaKeyID,
					Use:          "sig",
					Certificates: []*x509.Certificate{},
				},
			},
		},
	}

	for _, tt := range serveKeysTests {
		t.Run(tt.Name, func(t *testing.T) {
			s, _ := setupServer(t, exampleIssuer, tt.Keys)
			defer s.Close()

			reqURL := s.URL + "/openid/v1/jwks"

			resp, err := http.Get(reqURL)
			if err != nil {
				t.Fatalf("Get(%s) = %v, %v want: <response>, <nil>", reqURL, resp, err)
			}
			defer resp.Body.Close()

			if resp.StatusCode != http.StatusOK {
				t.Errorf("Get(%s) = %v, _ want: %v, _", reqURL, resp.StatusCode, http.StatusOK)
			}

			if got, want := resp.Header.Get("Content-Type"), "application/jwk-set+json"; got != want {
				t.Errorf("Get(%s) Content-Type = %q, _ want: %q, _", reqURL, got, want)
			}
			if got, want := resp.Header.Get("Cache-Control"), "public, max-age=3600"; got != want {
				t.Errorf("Get(%s) Cache-Control = %q, _ want: %q, _", reqURL, got, want)
			}

			ks := &jose.JSONWebKeySet{}
			if err := json.NewDecoder(resp.Body).Decode(ks); err != nil {
				t.Fatalf("Decode(_) = %v, want: <nil>", err)
			}

			bigIntComparer := cmp.Comparer(
				func(x, y *big.Int) bool {
					return x.Cmp(y) == 0
				})
			if !cmp.Equal(tt.WantKeys, ks.Keys, bigIntComparer) {
				t.Errorf("unexpected diff in JWKS keys (-want, +got): %v",
					cmp.Diff(tt.WantKeys, ks.Keys, bigIntComparer))
			}
		})
	}
}

func TestURLBoundaries(t *testing.T) {
	s, _ := setupServer(t, exampleIssuer, defaultKeys)
	defer s.Close()

	for _, tt := range []struct {
		Name   string
		Path   string
		WantOK bool
	}{
		{"OIDC config path", "/.well-known/openid-configuration", true},
		{"JWKS path", "/openid/v1/jwks", true},
		{"well-known", "/.well-known", false},
		{"subpath", "/openid/v1/jwks/foo", false},
		{"query", "/openid/v1/jwks?format=yaml", true},
		{"fragment", "/openid/v1/jwks#issuer", true},
	} {
		t.Run(tt.Name, func(t *testing.T) {
			resp, err := http.Get(s.URL + tt.Path)
			if err != nil {
				t.Fatal(err)
			}

			if tt.WantOK && (resp.StatusCode != http.StatusOK) {
				t.Errorf("Get(%v)= %v, want %v", tt.Path, resp.StatusCode, http.StatusOK)
			}
			if !tt.WantOK && (resp.StatusCode != http.StatusNotFound) {
				t.Errorf("Get(%v)= %v, want %v", tt.Path, resp.StatusCode, http.StatusNotFound)
			}
		})
	}
}

func TestNewOpenIDMetadata(t *testing.T) {
	cases := []struct {
		name            string
		issuerURL       string
		jwksURI         string
		externalAddress string
		keys            []interface{}
		wantConfig      string
		wantKeyset      string
		err             bool
	}{
		{
			name:       "valid inputs",
			issuerURL:  exampleIssuer,
			jwksURI:    exampleIssuer + serviceaccount.JWKSPath,
			keys:       defaultKeys,
			wantConfig: `{"issuer":"https://issuer.example.com","jwks_uri":"https://issuer.example.com/openid/v1/jwks","response_types_supported":["id_token"],"subject_types_supported":["public"],"id_token_signing_alg_values_supported":["ES256","RS256"]}`,
			wantKeyset: `{"keys":[{"use":"sig","kty":"RSA","kid":"JHJehTTTZlsspKHT-GaJxK7Kd1NQgZJu3fyK6K_QDYU","alg":"RS256","n":"249XwEo9k4tM8fMxV7zxOhcrP-WvXn917koM5Qr2ZXs4vo26e4ytdlrV0bQ9SlcLpQVSYjIxNfhTZdDt-ecIzshKuv1gKIxbbLQMOuK1eA_4HALyEkFgmS_tleLJrhc65tKPMGD-pKQ_xhmzRuCG51RoiMgbQxaCyYxGfNLpLAZK9L0Tctv9a0mJmGIYnIOQM4kC1A1I1n3EsXMWmeJUj7OTh_AjjCnMnkgvKT2tpKxYQ59PgDgU8Ssc7RDSmSkLxnrv-OrN80j6xrw0OjEiB4Ycr0PqfzZcvy8efTtFQ_Jnc4Bp1zUtFXt7-QeevePtQ2EcyELXE0i63T1CujRMWw","e":"AQAB"},{"use":"sig","kty":"EC","kid":"SoABiieYuNx4UdqYvZRVeuC6SihxgLrhLy9peHMHpTc","crv":"P-256","alg":"ES256","x":"H6cuzP8XuD5wal6wf9M6xDljTOPLX2i8uIp_C_ASqiI","y":"BlHnikLV9PyEd6gl8k4T_3Wwoh6xd79XLoQTh2PAi1Y"}]}`,
		},
		{
			name:      "valid inputs, default JWKSURI to external address",
			issuerURL: exampleIssuer,
			jwksURI:   "",
			// We expect host + port, no scheme, when API server calculates ExternalAddress.
			externalAddress: "192.0.2.1:80",
			keys:            defaultKeys,
			wantConfig:      `{"issuer":"https://issuer.example.com","jwks_uri":"https://192.0.2.1:80/openid/v1/jwks","response_types_supported":["id_token"],"subject_types_supported":["public"],"id_token_signing_alg_values_supported":["ES256","RS256"]}`,
			wantKeyset:      `{"keys":[{"use":"sig","kty":"RSA","kid":"JHJehTTTZlsspKHT-GaJxK7Kd1NQgZJu3fyK6K_QDYU","alg":"RS256","n":"249XwEo9k4tM8fMxV7zxOhcrP-WvXn917koM5Qr2ZXs4vo26e4ytdlrV0bQ9SlcLpQVSYjIxNfhTZdDt-ecIzshKuv1gKIxbbLQMOuK1eA_4HALyEkFgmS_tleLJrhc65tKPMGD-pKQ_xhmzRuCG51RoiMgbQxaCyYxGfNLpLAZK9L0Tctv9a0mJmGIYnIOQM4kC1A1I1n3EsXMWmeJUj7OTh_AjjCnMnkgvKT2tpKxYQ59PgDgU8Ssc7RDSmSkLxnrv-OrN80j6xrw0OjEiB4Ycr0PqfzZcvy8efTtFQ_Jnc4Bp1zUtFXt7-QeevePtQ2EcyELXE0i63T1CujRMWw","e":"AQAB"},{"use":"sig","kty":"EC","kid":"SoABiieYuNx4UdqYvZRVeuC6SihxgLrhLy9peHMHpTc","crv":"P-256","alg":"ES256","x":"H6cuzP8XuD5wal6wf9M6xDljTOPLX2i8uIp_C_ASqiI","y":"BlHnikLV9PyEd6gl8k4T_3Wwoh6xd79XLoQTh2PAi1Y"}]}`,
		},
		{
			name:       "valid inputs, IP addresses instead of domains",
			issuerURL:  "https://192.0.2.1:80",
			jwksURI:    "https://192.0.2.1:80" + serviceaccount.JWKSPath,
			keys:       defaultKeys,
			wantConfig: `{"issuer":"https://192.0.2.1:80","jwks_uri":"https://192.0.2.1:80/openid/v1/jwks","response_types_supported":["id_token"],"subject_types_supported":["public"],"id_token_signing_alg_values_supported":["ES256","RS256"]}`,
			wantKeyset: `{"keys":[{"use":"sig","kty":"RSA","kid":"JHJehTTTZlsspKHT-GaJxK7Kd1NQgZJu3fyK6K_QDYU","alg":"RS256","n":"249XwEo9k4tM8fMxV7zxOhcrP-WvXn917koM5Qr2ZXs4vo26e4ytdlrV0bQ9SlcLpQVSYjIxNfhTZdDt-ecIzshKuv1gKIxbbLQMOuK1eA_4HALyEkFgmS_tleLJrhc65tKPMGD-pKQ_xhmzRuCG51RoiMgbQxaCyYxGfNLpLAZK9L0Tctv9a0mJmGIYnIOQM4kC1A1I1n3EsXMWmeJUj7OTh_AjjCnMnkgvKT2tpKxYQ59PgDgU8Ssc7RDSmSkLxnrv-OrN80j6xrw0OjEiB4Ycr0PqfzZcvy8efTtFQ_Jnc4Bp1zUtFXt7-QeevePtQ2EcyELXE0i63T1CujRMWw","e":"AQAB"},{"use":"sig","kty":"EC","kid":"SoABiieYuNx4UdqYvZRVeuC6SihxgLrhLy9peHMHpTc","crv":"P-256","alg":"ES256","x":"H6cuzP8XuD5wal6wf9M6xDljTOPLX2i8uIp_C_ASqiI","y":"BlHnikLV9PyEd6gl8k4T_3Wwoh6xd79XLoQTh2PAi1Y"}]}`,
		},
		{
			name:       "response only contains public keys, even when private keys are provided",
			issuerURL:  exampleIssuer,
			jwksURI:    exampleIssuer + serviceaccount.JWKSPath,
			keys:       []interface{}{getPrivateKey(rsaPrivateKey), getPrivateKey(ecdsaPrivateKey)},
			wantConfig: `{"issuer":"https://issuer.example.com","jwks_uri":"https://issuer.example.com/openid/v1/jwks","response_types_supported":["id_token"],"subject_types_supported":["public"],"id_token_signing_alg_values_supported":["ES256","RS256"]}`,
			wantKeyset: `{"keys":[{"use":"sig","kty":"RSA","kid":"JHJehTTTZlsspKHT-GaJxK7Kd1NQgZJu3fyK6K_QDYU","alg":"RS256","n":"249XwEo9k4tM8fMxV7zxOhcrP-WvXn917koM5Qr2ZXs4vo26e4ytdlrV0bQ9SlcLpQVSYjIxNfhTZdDt-ecIzshKuv1gKIxbbLQMOuK1eA_4HALyEkFgmS_tleLJrhc65tKPMGD-pKQ_xhmzRuCG51RoiMgbQxaCyYxGfNLpLAZK9L0Tctv9a0mJmGIYnIOQM4kC1A1I1n3EsXMWmeJUj7OTh_AjjCnMnkgvKT2tpKxYQ59PgDgU8Ssc7RDSmSkLxnrv-OrN80j6xrw0OjEiB4Ycr0PqfzZcvy8efTtFQ_Jnc4Bp1zUtFXt7-QeevePtQ2EcyELXE0i63T1CujRMWw","e":"AQAB"},{"use":"sig","kty":"EC","kid":"SoABiieYuNx4UdqYvZRVeuC6SihxgLrhLy9peHMHpTc","crv":"P-256","alg":"ES256","x":"H6cuzP8XuD5wal6wf9M6xDljTOPLX2i8uIp_C_ASqiI","y":"BlHnikLV9PyEd6gl8k4T_3Wwoh6xd79XLoQTh2PAi1Y"}]}`,
		},
		{
			name:      "issuer missing https",
			issuerURL: "http://issuer.example.com",
			jwksURI:   exampleIssuer + serviceaccount.JWKSPath,
			keys:      defaultKeys,
			err:       true,
		},
		{
			name:      "issuer missing scheme",
			issuerURL: "issuer.example.com",
			jwksURI:   exampleIssuer + serviceaccount.JWKSPath,
			keys:      defaultKeys,
			err:       true,
		},
		{
			name:      "issuer includes query",
			issuerURL: "https://issuer.example.com?foo=bar",
			jwksURI:   exampleIssuer + serviceaccount.JWKSPath,
			keys:      defaultKeys,
			err:       true,
		},
		{
			name:      "issuer includes fragment",
			issuerURL: "https://issuer.example.com#baz",
			jwksURI:   exampleIssuer + serviceaccount.JWKSPath,
			keys:      defaultKeys,
			err:       true,
		},
		{
			name:      "issuer includes query and fragment",
			issuerURL: "https://issuer.example.com?foo=bar#baz",
			jwksURI:   exampleIssuer + serviceaccount.JWKSPath,
			keys:      defaultKeys,
			err:       true,
		},
		{
			name:      "issuer is not a valid URL",
			issuerURL: "issuer",
			jwksURI:   exampleIssuer + serviceaccount.JWKSPath,
			keys:      defaultKeys,
			err:       true,
		},
		{
			name:      "jwks missing https",
			issuerURL: exampleIssuer,
			jwksURI:   "http://issuer.example.com" + serviceaccount.JWKSPath,
			keys:      defaultKeys,
			err:       true,
		},
		{
			name:      "jwks missing scheme",
			issuerURL: exampleIssuer,
			jwksURI:   "issuer.example.com" + serviceaccount.JWKSPath,
			keys:      defaultKeys,
			err:       true,
		},
		{
			name:      "jwks is not a valid URL",
			issuerURL: exampleIssuer,
			jwksURI:   "issuer" + serviceaccount.JWKSPath,
			keys:      defaultKeys,
			err:       true,
		},
		{
			name:            "external address also has a scheme",
			issuerURL:       exampleIssuer,
			externalAddress: "https://192.0.2.1:80",
			keys:            defaultKeys,
			err:             true,
		},
		{
			name:      "missing external address and jwks",
			issuerURL: exampleIssuer,
			keys:      defaultKeys,
			err:       true,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			md, err := serviceaccount.NewOpenIDMetadata(tc.issuerURL, tc.jwksURI, tc.externalAddress, tc.keys)
			if tc.err {
				if err == nil {
					t.Fatalf("got <nil>, want error")
				}
				return
			} else if !tc.err && err != nil {
				t.Fatalf("got error %v, want <nil>", err)
			}

			config := string(md.ConfigJSON)
			keyset := string(md.PublicKeysetJSON)
			if config != tc.wantConfig {
				t.Errorf("got metadata %s, want %s", config, tc.wantConfig)
			}
			if keyset != tc.wantKeyset {
				t.Errorf("got keyset %s, want %s", keyset, tc.wantKeyset)
			}
		})
	}
}
