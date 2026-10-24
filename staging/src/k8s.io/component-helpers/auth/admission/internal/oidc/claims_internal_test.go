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
	"encoding/base64"
	"testing"
)

func TestParseUnverifiedClaimsRequiresCompactJWT(t *testing.T) {
	payload := base64.RawURLEncoding.EncodeToString([]byte(`{"iss":"https://issuer.test"}`))
	invalidJSON := base64.RawURLEncoding.EncodeToString([]byte(`{"iss":`))
	for _, tc := range []struct {
		name, token, issuer string
		wantErr             bool
	}{
		{name: "compact payload", token: "header." + payload + ".signature", issuer: "https://issuer.test"},
		{name: "raw JSON", token: ` {"iss":"https://issuer.test"}`, wantErr: true},
		{name: "two segments", token: "header." + payload, wantErr: true},
		{name: "four segments", token: "header." + payload + ".signature.extra", wantErr: true},
		{name: "invalid base64", token: "header.!.signature", wantErr: true},
		{name: "invalid JSON payload", token: "header." + invalidJSON + ".signature", wantErr: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			claims, err := parseUnverifiedClaims(tc.token)
			if (err != nil) != tc.wantErr {
				t.Fatalf("parseUnverifiedClaims error = %v, wantErr %t", err, tc.wantErr)
			}
			if err == nil && claims.Issuer != tc.issuer {
				t.Fatalf("unverified issuer = %q, want %q", claims.Issuer, tc.issuer)
			}
		})
	}
}
