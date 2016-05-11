/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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
	"io/ioutil"
	"os"
	"path"
	"testing"

	"github.com/coreos/go-oidc/jose"

	"k8s.io/kubernetes/pkg/util/diff"
	oidctesting "k8s.io/kubernetes/plugin/pkg/auth/authenticator/token/oidc/testing"
)

func TestNewOIDCAuthProvider(t *testing.T) {
	cert := path.Join(os.TempDir(), "oidc-cert")
	key := path.Join(os.TempDir(), "oidc-key")

	defer os.Remove(cert)
	defer os.Remove(key)

	oidctesting.GenerateSelfSignedCert(t, "127.0.0.1", cert, key)
	op := oidctesting.NewOIDCProvider(t)
	srv, err := op.ServeTLSWithKeyPair(cert, key)
	op.AddMinimalProviderConfig(srv)
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

	return diff.ObjectDiff(a, b)
}
