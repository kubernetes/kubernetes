/*
Copyright 2023 The Kubernetes Authors.

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

package validation

import (
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"encoding/pem"
	"testing"

	"github.com/google/go-cmp/cmp"

	"k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/validation/field"
	api "k8s.io/apiserver/pkg/apis/apiserver"
	certutil "k8s.io/client-go/util/cert"
	"k8s.io/utils/pointer"
)

func TestValidateAuthenticationConfiguration(t *testing.T) {
	testCases := []struct {
		name string
		in   *api.AuthenticationConfiguration
		want string
	}{
		{
			name: "jwt authenticator is empty",
			in:   &api.AuthenticationConfiguration{},
			want: "jwt: Required value: at least one jwt is required",
		},
		{
			name: ">1 jwt authenticator",
			in: &api.AuthenticationConfiguration{
				JWT: []api.JWTAuthenticator{
					{Issuer: api.Issuer{URL: "https://issuer-url", Audiences: []string{"audience"}}},
					{Issuer: api.Issuer{URL: "https://issuer-url", Audiences: []string{"audience"}}},
				},
			},
			want: "jwt: Too many: 2: must have at most 1 items",
		},
		{
			name: "failed issuer validation",
			in: &api.AuthenticationConfiguration{
				JWT: []api.JWTAuthenticator{
					{
						Issuer: api.Issuer{
							URL:       "invalid-url",
							Audiences: []string{"audience"},
						},
						ClaimMappings: api.ClaimMappings{
							Username: api.PrefixedClaimOrExpression{
								Claim:  "claim",
								Prefix: pointer.String("prefix"),
							},
						},
					},
				},
			},
			want: `jwt[0].issuer.url: Invalid value: "invalid-url": URL scheme must be https`,
		},
		{
			name: "failed claimValidationRule validation",
			in: &api.AuthenticationConfiguration{
				JWT: []api.JWTAuthenticator{
					{
						Issuer: api.Issuer{
							URL:       "https://issuer-url",
							Audiences: []string{"audience"},
						},
						ClaimValidationRules: []api.ClaimValidationRule{
							{
								Claim:         "foo",
								RequiredValue: "bar",
							},
							{
								Claim:         "foo",
								RequiredValue: "baz",
							},
						},
						ClaimMappings: api.ClaimMappings{
							Username: api.PrefixedClaimOrExpression{
								Claim:  "claim",
								Prefix: pointer.String("prefix"),
							},
						},
					},
				},
			},
			want: `jwt[0].claimValidationRules[1].claim: Duplicate value: "foo"`,
		},
		{
			name: "failed claimMapping validation",
			in: &api.AuthenticationConfiguration{
				JWT: []api.JWTAuthenticator{
					{
						Issuer: api.Issuer{
							URL:       "https://issuer-url",
							Audiences: []string{"audience"},
						},
						ClaimValidationRules: []api.ClaimValidationRule{
							{
								Claim:         "foo",
								RequiredValue: "bar",
							},
						},
						ClaimMappings: api.ClaimMappings{
							Username: api.PrefixedClaimOrExpression{
								Prefix: pointer.String("prefix"),
							},
						},
					},
				},
			},
			want: "jwt[0].claimMappings.username.claim: Required value: claim name is required",
		},
		{
			name: "valid authentication configuration",
			in: &api.AuthenticationConfiguration{
				JWT: []api.JWTAuthenticator{
					{
						Issuer: api.Issuer{
							URL:       "https://issuer-url",
							Audiences: []string{"audience"},
						},
						ClaimValidationRules: []api.ClaimValidationRule{
							{
								Claim:         "foo",
								RequiredValue: "bar",
							},
						},
						ClaimMappings: api.ClaimMappings{
							Username: api.PrefixedClaimOrExpression{
								Claim:  "sub",
								Prefix: pointer.String("prefix"),
							},
						},
					},
				},
			},
			want: "",
		},
	}

	for _, tt := range testCases {
		t.Run(tt.name, func(t *testing.T) {
			got := ValidateAuthenticationConfiguration(tt.in).ToAggregate()
			if d := cmp.Diff(tt.want, errString(got)); d != "" {
				t.Fatalf("AuthenticationConfiguration validation mismatch (-want +got):\n%s", d)
			}
		})
	}
}

func TestValidateURL(t *testing.T) {
	fldPath := field.NewPath("issuer", "url")

	testCases := []struct {
		name string
		in   string
		want string
	}{
		{
			name: "url is empty",
			in:   "",
			want: "issuer.url: Required value: URL is required",
		},
		{
			name: "url parse error",
			in:   "https://issuer-url:invalid-port",
			want: `issuer.url: Invalid value: "https://issuer-url:invalid-port": parse "https://issuer-url:invalid-port": invalid port ":invalid-port" after host`,
		},
		{
			name: "url is not https",
			in:   "http://issuer-url",
			want: `issuer.url: Invalid value: "http://issuer-url": URL scheme must be https`,
		},
		{
			name: "url user info is not allowed",
			in:   "https://user:pass@issuer-url",
			want: `issuer.url: Invalid value: "https://user:pass@issuer-url": URL must not contain a username or password`,
		},
		{
			name: "url raw query is not allowed",
			in:   "https://issuer-url?query",
			want: `issuer.url: Invalid value: "https://issuer-url?query": URL must not contain a query`,
		},
		{
			name: "url fragment is not allowed",
			in:   "https://issuer-url#fragment",
			want: `issuer.url: Invalid value: "https://issuer-url#fragment": URL must not contain a fragment`,
		},
		{
			name: "valid url",
			in:   "https://issuer-url",
			want: "",
		},
	}

	for _, tt := range testCases {
		t.Run(tt.name, func(t *testing.T) {
			got := validateURL(tt.in, fldPath).ToAggregate()
			if d := cmp.Diff(tt.want, errString(got)); d != "" {
				t.Fatalf("URL validation mismatch (-want +got):\n%s", d)
			}
		})
	}
}

func TestValidateAudiences(t *testing.T) {
	fldPath := field.NewPath("issuer", "audiences")

	testCases := []struct {
		name string
		in   []string
		want string
	}{
		{
			name: "audiences is empty",
			in:   []string{},
			want: "issuer.audiences: Required value: at least one issuer.audiences is required",
		},
		{
			name: "at most one audiences is allowed",
			in:   []string{"audience1", "audience2"},
			want: "issuer.audiences: Too many: 2: must have at most 1 items",
		},
		{
			name: "audience is empty",
			in:   []string{""},
			want: "issuer.audiences[0]: Required value: audience can't be empty",
		},
		{
			name: "valid audience",
			in:   []string{"audience"},
			want: "",
		},
	}

	for _, tt := range testCases {
		t.Run(tt.name, func(t *testing.T) {
			got := validateAudiences(tt.in, fldPath).ToAggregate()
			if d := cmp.Diff(tt.want, errString(got)); d != "" {
				t.Fatalf("Audiences validation mismatch (-want +got):\n%s", d)
			}
		})
	}
}

func TestValidateCertificateAuthority(t *testing.T) {
	fldPath := field.NewPath("issuer", "certificateAuthority")

	testCases := []struct {
		name string
		in   func() string
		want string
	}{
		{
			name: "invalid certificate authority",
			in:   func() string { return "invalid" },
			want: `issuer.certificateAuthority: Invalid value: "<omitted>": data does not contain any valid RSA or ECDSA certificates`,
		},
		{
			name: "certificate authority is empty",
			in:   func() string { return "" },
			want: "",
		},
		{
			name: "valid certificate authority",
			in: func() string {
				caPrivateKey, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
				if err != nil {
					t.Fatal(err)
				}
				caCert, err := certutil.NewSelfSignedCACert(certutil.Config{CommonName: "test-ca"}, caPrivateKey)
				if err != nil {
					t.Fatal(err)
				}
				return string(pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: caCert.Raw}))
			},
			want: "",
		},
	}

	for _, tt := range testCases {
		t.Run(tt.name, func(t *testing.T) {
			got := validateCertificateAuthority(tt.in(), fldPath).ToAggregate()
			if d := cmp.Diff(tt.want, errString(got)); d != "" {
				t.Fatalf("CertificateAuthority validation mismatch (-want +got):\n%s", d)
			}
		})
	}
}

func TestClaimValidationRules(t *testing.T) {
	fldPath := field.NewPath("issuer", "claimValidationRules")

	testCases := []struct {
		name string
		in   []api.ClaimValidationRule
		want string
	}{
		{
			name: "claim validation rule claim is empty",
			in:   []api.ClaimValidationRule{{Claim: ""}},
			want: "issuer.claimValidationRules[0].claim: Required value: claim name is required",
		},
		{
			name: "duplicate claim",
			in: []api.ClaimValidationRule{{
				Claim: "claim", RequiredValue: "value1"},
				{Claim: "claim", RequiredValue: "value2"},
			},
			want: `issuer.claimValidationRules[1].claim: Duplicate value: "claim"`,
		},
		{
			name: "valid claim validation rule",
			in:   []api.ClaimValidationRule{{Claim: "claim", RequiredValue: "value"}},
			want: "",
		},
		{
			name: "valid claim validation rule with multiple rules",
			in: []api.ClaimValidationRule{
				{Claim: "claim1", RequiredValue: "value1"},
				{Claim: "claim2", RequiredValue: "value2"},
			},
			want: "",
		},
	}

	for _, tt := range testCases {
		t.Run(tt.name, func(t *testing.T) {
			got := validateClaimValidationRules(tt.in, fldPath).ToAggregate()
			if d := cmp.Diff(tt.want, errString(got)); d != "" {
				t.Fatalf("ClaimValidationRules validation mismatch (-want +got):\n%s", d)
			}
		})
	}
}

func TestValidateClaimMappings(t *testing.T) {
	fldPath := field.NewPath("issuer", "claimMappings")

	testCases := []struct {
		name string
		in   api.ClaimMappings
		want string
	}{
		{
			name: "username claim is empty",
			in:   api.ClaimMappings{Username: api.PrefixedClaimOrExpression{Claim: "", Prefix: pointer.String("prefix")}},
			want: "issuer.claimMappings.username.claim: Required value: claim name is required",
		},
		{
			name: "username prefix is empty",
			in:   api.ClaimMappings{Username: api.PrefixedClaimOrExpression{Claim: "claim"}},
			want: "issuer.claimMappings.username.prefix: Required value: prefix is required",
		},
		{
			name: "groups prefix is empty",
			in: api.ClaimMappings{
				Username: api.PrefixedClaimOrExpression{Claim: "claim", Prefix: pointer.String("prefix")},
				Groups:   api.PrefixedClaimOrExpression{Claim: "claim"},
			},
			want: "issuer.claimMappings.groups.prefix: Required value: prefix is required when claim is set",
		},
		{
			name: "groups prefix set but claim is empty",
			in: api.ClaimMappings{
				Username: api.PrefixedClaimOrExpression{Claim: "claim", Prefix: pointer.String("prefix")},
				Groups:   api.PrefixedClaimOrExpression{Prefix: pointer.String("prefix")},
			},
			want: "issuer.claimMappings.groups.claim: Required value: non-empty claim name is required when prefix is set",
		},
		{
			name: "valid claim mappings",
			in: api.ClaimMappings{
				Username: api.PrefixedClaimOrExpression{Claim: "claim", Prefix: pointer.String("prefix")},
				Groups:   api.PrefixedClaimOrExpression{Claim: "claim", Prefix: pointer.String("prefix")},
			},
			want: "",
		},
	}

	for _, tt := range testCases {
		t.Run(tt.name, func(t *testing.T) {
			got := validateClaimMappings(tt.in, fldPath).ToAggregate()
			if d := cmp.Diff(tt.want, errString(got)); d != "" {
				t.Fatalf("ClaimMappings validation mismatch (-want +got):\n%s", d)
			}
		})
	}
}

func errString(errs errors.Aggregate) string {
	if errs != nil {
		return errs.Error()
	}
	return ""
}
