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
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation/field"
	api "k8s.io/apiserver/pkg/apis/apiserver"
	authenticationcel "k8s.io/apiserver/pkg/authentication/cel"
	"k8s.io/apiserver/pkg/cel/environment"
	"k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	certutil "k8s.io/client-go/util/cert"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/utils/pointer"
)

var (
	compiler = authenticationcel.NewCompiler(environment.MustBaseEnvSet(environment.DefaultCompatibilityVersion()))
)

func TestValidateAuthenticationConfiguration(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.StructuredAuthenticationConfiguration, true)()

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
			want: "jwt[0].claimMappings.username: Required value: claim or expression is required",
		},
		{
			name: "failed userValidationRule validation",
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
						UserValidationRules: []api.UserValidationRule{
							{Expression: "user.username == 'foo'"},
							{Expression: "user.username == 'foo'"},
						},
					},
				},
			},
			want: `jwt[0].userValidationRules[1].expression: Duplicate value: "user.username == 'foo'"`,
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
	audienceMatchPolicyFldPath := field.NewPath("issuer", "audienceMatchPolicy")

	testCases := []struct {
		name        string
		in          []string
		matchPolicy string
		want        string
	}{
		{
			name: "audiences is empty",
			in:   []string{},
			want: "issuer.audiences: Required value: at least one issuer.audiences is required",
		},
		{
			name: "audience is empty",
			in:   []string{""},
			want: "issuer.audiences[0]: Required value: audience can't be empty",
		},
		{
			name:        "invalid match policy with single audience",
			in:          []string{"audience"},
			matchPolicy: "MatchExact",
			want:        `issuer.audienceMatchPolicy: Invalid value: "MatchExact": audienceMatchPolicy must be empty or MatchAny for single audience`,
		},
		{
			name: "valid audience",
			in:   []string{"audience"},
			want: "",
		},
		{
			name:        "valid audience with MatchAny policy",
			in:          []string{"audience"},
			matchPolicy: "MatchAny",
			want:        "",
		},
		{
			name:        "duplicate audience",
			in:          []string{"audience", "audience"},
			matchPolicy: "MatchAny",
			want:        `issuer.audiences[1]: Duplicate value: "audience"`,
		},
		{
			name: "match policy not set with multiple audiences",
			in:   []string{"audience1", "audience2"},
			want: `issuer.audienceMatchPolicy: Invalid value: "": audienceMatchPolicy must be MatchAny for multiple audiences`,
		},
		{
			name:        "valid multiple audiences",
			in:          []string{"audience1", "audience2"},
			matchPolicy: "MatchAny",
			want:        "",
		},
	}

	for _, tt := range testCases {
		t.Run(tt.name, func(t *testing.T) {
			got := validateAudiences(tt.in, api.AudienceMatchPolicyType(tt.matchPolicy), fldPath, audienceMatchPolicyFldPath).ToAggregate()
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

func TestValidateClaimValidationRules(t *testing.T) {
	fldPath := field.NewPath("issuer", "claimValidationRules")

	testCases := []struct {
		name                          string
		in                            []api.ClaimValidationRule
		structuredAuthnFeatureEnabled bool
		want                          string
		wantCELMapper                 bool
	}{
		{
			name:                          "claim and expression are empty, structured authn feature enabled",
			in:                            []api.ClaimValidationRule{{}},
			structuredAuthnFeatureEnabled: true,
			want:                          "issuer.claimValidationRules[0]: Required value: claim or expression is required",
		},
		{
			name: "claim and expression are set",
			in: []api.ClaimValidationRule{
				{Claim: "claim", Expression: "expression"},
			},
			structuredAuthnFeatureEnabled: true,
			want:                          `issuer.claimValidationRules[0]: Invalid value: "claim": claim and expression can't both be set`,
		},
		{
			name: "message set when claim is set",
			in: []api.ClaimValidationRule{
				{Claim: "claim", Message: "message"},
			},
			structuredAuthnFeatureEnabled: true,
			want:                          `issuer.claimValidationRules[0].message: Invalid value: "message": message can't be set when claim is set`,
		},
		{
			name: "requiredValue set when expression is set",
			in: []api.ClaimValidationRule{
				{Expression: "claims.foo == 'bar'", RequiredValue: "value"},
			},
			structuredAuthnFeatureEnabled: true,
			want:                          `issuer.claimValidationRules[0].requiredValue: Invalid value: "value": requiredValue can't be set when expression is set`,
		},
		{
			name: "duplicate claim",
			in: []api.ClaimValidationRule{
				{Claim: "claim"},
				{Claim: "claim"},
			},
			structuredAuthnFeatureEnabled: true,
			want:                          `issuer.claimValidationRules[1].claim: Duplicate value: "claim"`,
		},
		{
			name: "duplicate expression",
			in: []api.ClaimValidationRule{
				{Expression: "claims.foo == 'bar'"},
				{Expression: "claims.foo == 'bar'"},
			},
			structuredAuthnFeatureEnabled: true,
			want:                          `issuer.claimValidationRules[1].expression: Duplicate value: "claims.foo == 'bar'"`,
		},
		{
			name: "expression set when structured authn feature is disabled",
			in: []api.ClaimValidationRule{
				{Expression: "claims.foo == 'bar'"},
			},
			structuredAuthnFeatureEnabled: false,
			want:                          `issuer.claimValidationRules[0].expression: Invalid value: "claims.foo == 'bar'": expression is not supported when StructuredAuthenticationConfiguration feature gate is disabled`,
		},
		{
			name: "CEL expression compilation error",
			in: []api.ClaimValidationRule{
				{Expression: "foo.bar"},
			},
			structuredAuthnFeatureEnabled: true,
			want: `issuer.claimValidationRules[0].expression: Invalid value: "foo.bar": compilation failed: ERROR: <input>:1:1: undeclared reference to 'foo' (in container '')
 | foo.bar
 | ^`,
		},
		{
			name: "expression does not evaluate to bool",
			in: []api.ClaimValidationRule{
				{Expression: "claims.foo"},
			},
			structuredAuthnFeatureEnabled: true,
			want:                          `issuer.claimValidationRules[0].expression: Invalid value: "claims.foo": must evaluate to bool`,
		},
		{
			name: "valid claim validation rule with expression",
			in: []api.ClaimValidationRule{
				{Expression: "claims.foo == 'bar'"},
			},
			structuredAuthnFeatureEnabled: true,
			want:                          "",
			wantCELMapper:                 true,
		},
		{
			name: "valid claim validation rule with multiple rules",
			in: []api.ClaimValidationRule{
				{Claim: "claim1", RequiredValue: "value1"},
				{Claim: "claim2", RequiredValue: "value2"},
			},
			structuredAuthnFeatureEnabled: true,
			want:                          "",
		},
	}

	for _, tt := range testCases {
		t.Run(tt.name, func(t *testing.T) {
			celMapper := &authenticationcel.CELMapper{}
			got := validateClaimValidationRules(compiler, celMapper, tt.in, fldPath, tt.structuredAuthnFeatureEnabled).ToAggregate()
			if d := cmp.Diff(tt.want, errString(got)); d != "" {
				t.Fatalf("ClaimValidationRules validation mismatch (-want +got):\n%s", d)
			}
			if tt.wantCELMapper && celMapper.ClaimValidationRules == nil {
				t.Fatalf("ClaimValidationRules validation mismatch: CELMapper.ClaimValidationRules is nil")
			}
		})
	}
}

func TestValidateClaimMappings(t *testing.T) {
	fldPath := field.NewPath("issuer", "claimMappings")

	testCases := []struct {
		name                          string
		in                            api.ClaimMappings
		structuredAuthnFeatureEnabled bool
		want                          string
		wantCELMapper                 bool
	}{
		{
			name: "username expression and claim are set",
			in: api.ClaimMappings{
				Username: api.PrefixedClaimOrExpression{
					Claim:      "claim",
					Expression: "claims.username",
				},
			},
			structuredAuthnFeatureEnabled: true,
			want:                          `issuer.claimMappings.username: Invalid value: "": claim and expression can't both be set`,
		},
		{
			name:                          "username expression and claim are empty",
			in:                            api.ClaimMappings{Username: api.PrefixedClaimOrExpression{}},
			structuredAuthnFeatureEnabled: true,
			want:                          "issuer.claimMappings.username: Required value: claim or expression is required",
		},
		{
			name: "username prefix set when expression is set",
			in: api.ClaimMappings{
				Username: api.PrefixedClaimOrExpression{
					Expression: "claims.username",
					Prefix:     pointer.String("prefix"),
				},
			},
			structuredAuthnFeatureEnabled: true,
			want:                          `issuer.claimMappings.username.prefix: Invalid value: "prefix": prefix can't be set when expression is set`,
		},
		{
			name: "username prefix is nil when claim is set",
			in: api.ClaimMappings{
				Username: api.PrefixedClaimOrExpression{
					Claim: "claim",
				},
			},
			structuredAuthnFeatureEnabled: true,
			want:                          `issuer.claimMappings.username.prefix: Required value: prefix is required when claim is set. It can be set to an empty string to disable prefixing`,
		},
		{
			name: "username expression is invalid",
			in: api.ClaimMappings{
				Username: api.PrefixedClaimOrExpression{
					Expression: "foo.bar",
				},
			},
			structuredAuthnFeatureEnabled: true,
			want: `issuer.claimMappings.username.expression: Invalid value: "foo.bar": compilation failed: ERROR: <input>:1:1: undeclared reference to 'foo' (in container '')
 | foo.bar
 | ^`,
		},
		{
			name: "groups expression and claim are set",
			in: api.ClaimMappings{
				Username: api.PrefixedClaimOrExpression{
					Claim:  "claim",
					Prefix: pointer.String("prefix"),
				},
				Groups: api.PrefixedClaimOrExpression{
					Claim:      "claim",
					Expression: "claims.groups",
				},
			},
			structuredAuthnFeatureEnabled: true,
			want:                          `issuer.claimMappings.groups: Invalid value: "": claim and expression can't both be set`,
		},
		{
			name: "groups prefix set when expression is set",
			in: api.ClaimMappings{
				Username: api.PrefixedClaimOrExpression{
					Claim:  "claim",
					Prefix: pointer.String("prefix"),
				},
				Groups: api.PrefixedClaimOrExpression{
					Expression: "claims.groups",
					Prefix:     pointer.String("prefix"),
				},
			},
			structuredAuthnFeatureEnabled: true,
			want:                          `issuer.claimMappings.groups.prefix: Invalid value: "prefix": prefix can't be set when expression is set`,
		},
		{
			name: "groups prefix is nil when claim is set",
			in: api.ClaimMappings{
				Username: api.PrefixedClaimOrExpression{
					Claim:  "claim",
					Prefix: pointer.String("prefix"),
				},
				Groups: api.PrefixedClaimOrExpression{
					Claim: "claim",
				},
			},
			structuredAuthnFeatureEnabled: true,
			want:                          `issuer.claimMappings.groups.prefix: Required value: prefix is required when claim is set. It can be set to an empty string to disable prefixing`,
		},
		{
			name: "groups expression is invalid",
			in: api.ClaimMappings{
				Username: api.PrefixedClaimOrExpression{
					Claim:  "claim",
					Prefix: pointer.String("prefix"),
				},
				Groups: api.PrefixedClaimOrExpression{
					Expression: "foo.bar",
				},
			},
			structuredAuthnFeatureEnabled: true,
			want: `issuer.claimMappings.groups.expression: Invalid value: "foo.bar": compilation failed: ERROR: <input>:1:1: undeclared reference to 'foo' (in container '')
 | foo.bar
 | ^`,
		},
		{
			name: "uid claim and expression are set",
			in: api.ClaimMappings{
				Username: api.PrefixedClaimOrExpression{
					Claim:  "claim",
					Prefix: pointer.String("prefix"),
				},
				UID: api.ClaimOrExpression{
					Claim:      "claim",
					Expression: "claims.uid",
				},
			},
			structuredAuthnFeatureEnabled: true,
			want:                          `issuer.claimMappings.uid: Invalid value: "": claim and expression can't both be set`,
		},
		{
			name: "uid expression is invalid",
			in: api.ClaimMappings{
				Username: api.PrefixedClaimOrExpression{
					Claim:  "claim",
					Prefix: pointer.String("prefix"),
				},
				UID: api.ClaimOrExpression{
					Expression: "foo.bar",
				},
			},
			structuredAuthnFeatureEnabled: true,
			want: `issuer.claimMappings.uid.expression: Invalid value: "foo.bar": compilation failed: ERROR: <input>:1:1: undeclared reference to 'foo' (in container '')
 | foo.bar
 | ^`,
		},
		{
			name: "extra mapping key is empty",
			in: api.ClaimMappings{
				Username: api.PrefixedClaimOrExpression{
					Claim:  "claim",
					Prefix: pointer.String("prefix"),
				},
				Extra: []api.ExtraMapping{
					{Key: "", ValueExpression: "claims.extra"},
				},
			},
			structuredAuthnFeatureEnabled: true,
			want:                          `issuer.claimMappings.extra[0].key: Required value`,
		},
		{
			name: "extra mapping value expression is empty",
			in: api.ClaimMappings{
				Username: api.PrefixedClaimOrExpression{
					Claim:  "claim",
					Prefix: pointer.String("prefix"),
				},
				Extra: []api.ExtraMapping{
					{Key: "example.org/foo", ValueExpression: ""},
				},
			},
			structuredAuthnFeatureEnabled: true,
			want:                          `issuer.claimMappings.extra[0].valueExpression: Required value: valueExpression is required`,
		},
		{
			name: "extra mapping value expression is invalid",
			in: api.ClaimMappings{
				Username: api.PrefixedClaimOrExpression{
					Claim:  "claim",
					Prefix: pointer.String("prefix"),
				},
				Extra: []api.ExtraMapping{
					{Key: "example.org/foo", ValueExpression: "foo.bar"},
				},
			},
			structuredAuthnFeatureEnabled: true,
			want: `issuer.claimMappings.extra[0].valueExpression: Invalid value: "foo.bar": compilation failed: ERROR: <input>:1:1: undeclared reference to 'foo' (in container '')
 | foo.bar
 | ^`,
		},
		{
			name: "username expression is invalid when structured authn feature is disabled",
			in: api.ClaimMappings{
				Username: api.PrefixedClaimOrExpression{
					Expression: "foo.bar",
				},
			},
			structuredAuthnFeatureEnabled: false,
			want: `[issuer.claimMappings.username.expression: Invalid value: "foo.bar": expression is not supported when StructuredAuthenticationConfiguration feature gate is disabled, issuer.claimMappings.username.expression: Invalid value: "foo.bar": compilation failed: ERROR: <input>:1:1: undeclared reference to 'foo' (in container '')
 | foo.bar
 | ^]`,
		},
		{
			name: "groups expression is invalid when structured authn feature is disabled",
			in: api.ClaimMappings{
				Username: api.PrefixedClaimOrExpression{
					Claim:  "claim",
					Prefix: pointer.String("prefix"),
				},
				Groups: api.PrefixedClaimOrExpression{
					Expression: "foo.bar",
				},
			},
			structuredAuthnFeatureEnabled: false,
			want: `[issuer.claimMappings.groups.expression: Invalid value: "foo.bar": expression is not supported when StructuredAuthenticationConfiguration feature gate is disabled, issuer.claimMappings.groups.expression: Invalid value: "foo.bar": compilation failed: ERROR: <input>:1:1: undeclared reference to 'foo' (in container '')
 | foo.bar
 | ^]`,
		},
		{
			name: "uid expression is invalid when structured authn feature is disabled",
			in: api.ClaimMappings{
				Username: api.PrefixedClaimOrExpression{
					Claim:  "claim",
					Prefix: pointer.String("prefix"),
				},
				UID: api.ClaimOrExpression{
					Expression: "foo.bar",
				},
			},
			structuredAuthnFeatureEnabled: false,
			want: `[issuer.claimMappings.uid: Invalid value: "": uid claim mapping is not supported when StructuredAuthenticationConfiguration feature gate is disabled, issuer.claimMappings.uid.expression: Invalid value: "foo.bar": compilation failed: ERROR: <input>:1:1: undeclared reference to 'foo' (in container '')
 | foo.bar
 | ^]`,
		},
		{
			name: "uid claim is invalid when structured authn feature is disabled",
			in: api.ClaimMappings{
				Username: api.PrefixedClaimOrExpression{
					Claim:  "claim",
					Prefix: pointer.String("prefix"),
				},
				UID: api.ClaimOrExpression{
					Claim: "claim",
				},
			},
			structuredAuthnFeatureEnabled: false,
			want:                          `issuer.claimMappings.uid: Invalid value: "": uid claim mapping is not supported when StructuredAuthenticationConfiguration feature gate is disabled`,
		},
		{
			name: "extra mapping is invalid when structured authn feature is disabled",
			in: api.ClaimMappings{
				Username: api.PrefixedClaimOrExpression{
					Claim:  "claim",
					Prefix: pointer.String("prefix"),
				},
				Extra: []api.ExtraMapping{
					{Key: "example.org/foo", ValueExpression: "claims.extra"},
				},
			},
			structuredAuthnFeatureEnabled: false,
			want:                          `issuer.claimMappings.extra: Invalid value: "": extra claim mapping is not supported when StructuredAuthenticationConfiguration feature gate is disabled`,
		},
		{
			name: "duplicate extra mapping key",
			in: api.ClaimMappings{
				Username: api.PrefixedClaimOrExpression{Expression: "claims.username"},
				Groups:   api.PrefixedClaimOrExpression{Expression: "claims.groups"},
				Extra: []api.ExtraMapping{
					{Key: "example.org/foo", ValueExpression: "claims.extra"},
					{Key: "example.org/foo", ValueExpression: "claims.extras"},
				},
			},
			structuredAuthnFeatureEnabled: true,
			want:                          `issuer.claimMappings.extra[1].key: Duplicate value: "example.org/foo"`,
		},
		{
			name: "extra mapping key is not domain prefix path",
			in: api.ClaimMappings{
				Username: api.PrefixedClaimOrExpression{Expression: "claims.username"},
				Groups:   api.PrefixedClaimOrExpression{Expression: "claims.groups"},
				Extra: []api.ExtraMapping{
					{Key: "foo", ValueExpression: "claims.extra"},
				},
			},
			structuredAuthnFeatureEnabled: true,
			want:                          `issuer.claimMappings.extra[0].key: Invalid value: "foo": must be a domain-prefixed path (such as "acme.io/foo")`,
		},
		{
			name: "extra mapping key is not lower case",
			in: api.ClaimMappings{
				Username: api.PrefixedClaimOrExpression{Expression: "claims.username"},
				Groups:   api.PrefixedClaimOrExpression{Expression: "claims.groups"},
				Extra: []api.ExtraMapping{
					{Key: "example.org/Foo", ValueExpression: "claims.extra"},
				},
			},
			structuredAuthnFeatureEnabled: true,
			want:                          `issuer.claimMappings.extra[0].key: Invalid value: "example.org/Foo": key must be lowercase`,
		},
		{
			name: "valid claim mappings",
			in: api.ClaimMappings{
				Username: api.PrefixedClaimOrExpression{Expression: "claims.username"},
				Groups:   api.PrefixedClaimOrExpression{Expression: "claims.groups"},
				UID:      api.ClaimOrExpression{Expression: "claims.uid"},
				Extra: []api.ExtraMapping{
					{Key: "example.org/foo", ValueExpression: "claims.extra"},
				},
			},
			structuredAuthnFeatureEnabled: true,
			wantCELMapper:                 true,
			want:                          "",
		},
	}

	for _, tt := range testCases {
		t.Run(tt.name, func(t *testing.T) {
			celMapper := &authenticationcel.CELMapper{}
			got := validateClaimMappings(compiler, celMapper, tt.in, fldPath, tt.structuredAuthnFeatureEnabled).ToAggregate()
			if d := cmp.Diff(tt.want, errString(got)); d != "" {
				fmt.Println(errString(got))
				t.Fatalf("ClaimMappings validation mismatch (-want +got):\n%s", d)
			}
			if tt.wantCELMapper {
				if len(tt.in.Username.Expression) > 0 && celMapper.Username == nil {
					t.Fatalf("ClaimMappings validation mismatch: CELMapper.Username is nil")
				}
				if len(tt.in.Groups.Expression) > 0 && celMapper.Groups == nil {
					t.Fatalf("ClaimMappings validation mismatch: CELMapper.Groups is nil")
				}
				if len(tt.in.UID.Expression) > 0 && celMapper.UID == nil {
					t.Fatalf("ClaimMappings validation mismatch: CELMapper.UID is nil")
				}
				if len(tt.in.Extra) > 0 && celMapper.Extra == nil {
					t.Fatalf("ClaimMappings validation mismatch: CELMapper.Extra is nil")
				}
			}
		})
	}
}

func TestValidateUserValidationRules(t *testing.T) {
	fldPath := field.NewPath("issuer", "userValidationRules")

	testCases := []struct {
		name                          string
		in                            []api.UserValidationRule
		structuredAuthnFeatureEnabled bool
		want                          string
		wantCELMapper                 bool
	}{
		{
			name:                          "user info validation rule, expression is empty",
			in:                            []api.UserValidationRule{{}},
			structuredAuthnFeatureEnabled: true,
			want:                          "issuer.userValidationRules[0].expression: Required value: expression is required",
		},
		{
			name: "duplicate expression",
			in: []api.UserValidationRule{
				{Expression: "user.username == 'foo'"},
				{Expression: "user.username == 'foo'"},
			},
			structuredAuthnFeatureEnabled: true,
			want:                          `issuer.userValidationRules[1].expression: Duplicate value: "user.username == 'foo'"`,
		},
		{
			name: "user validation rule is invalid when structured authn feature is disabled",
			in: []api.UserValidationRule{
				{Expression: "user.username == 'foo'"},
			},
			structuredAuthnFeatureEnabled: false,
			want:                          `issuer.userValidationRules: Invalid value: "": user validation rules are not supported when StructuredAuthenticationConfiguration feature gate is disabled`,
		},
		{
			name: "expression is invalid",
			in: []api.UserValidationRule{
				{Expression: "foo.bar"},
			},
			structuredAuthnFeatureEnabled: true,
			want: `issuer.userValidationRules[0].expression: Invalid value: "foo.bar": compilation failed: ERROR: <input>:1:1: undeclared reference to 'foo' (in container '')
 | foo.bar
 | ^`,
		},
		{
			name: "expression does not return bool",
			in: []api.UserValidationRule{
				{Expression: "user.username"},
			},
			structuredAuthnFeatureEnabled: true,
			want:                          `issuer.userValidationRules[0].expression: Invalid value: "user.username": must evaluate to bool`,
		},
		{
			name: "valid user info validation rule",
			in: []api.UserValidationRule{
				{Expression: "user.username == 'foo'"},
				{Expression: "!user.username.startsWith('system:')", Message: "username cannot used reserved system: prefix"},
			},
			structuredAuthnFeatureEnabled: true,
			want:                          "",
			wantCELMapper:                 true,
		},
	}

	for _, tt := range testCases {
		t.Run(tt.name, func(t *testing.T) {
			celMapper := &authenticationcel.CELMapper{}
			got := validateUserValidationRules(compiler, celMapper, tt.in, fldPath, tt.structuredAuthnFeatureEnabled).ToAggregate()
			if d := cmp.Diff(tt.want, errString(got)); d != "" {
				t.Fatalf("UserValidationRules validation mismatch (-want +got):\n%s", d)
			}
			if tt.wantCELMapper && celMapper.UserValidationRules == nil {
				t.Fatalf("UserValidationRules validation mismatch: CELMapper.UserValidationRules is nil")
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

type (
	test struct {
		name            string
		configuration   api.AuthorizationConfiguration
		expectedErrList field.ErrorList
		knownTypes      sets.String
		repeatableTypes sets.String
	}
)

func TestValidateAuthorizationConfiguration(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.StructuredAuthorizationConfiguration, true)()

	badKubeConfigFile := "../some/relative/path/kubeconfig"

	tempKubeConfigFile, err := os.CreateTemp("/tmp", "kubeconfig")
	if err != nil {
		t.Fatalf("failed to set up temp file: %v", err)
	}
	tempKubeConfigFilePath := tempKubeConfigFile.Name()
	defer os.Remove(tempKubeConfigFilePath)

	tests := []test{
		{
			name: "atleast one authorizer should be defined",
			configuration: api.AuthorizationConfiguration{
				Authorizers: []api.AuthorizerConfiguration{},
			},
			expectedErrList: field.ErrorList{field.Required(field.NewPath("authorizers"), "at least one authorization mode must be defined")},
			knownTypes:      sets.NewString(),
			repeatableTypes: sets.NewString(),
		},
		{
			name: "type and name are required if an authorizer is defined",
			configuration: api.AuthorizationConfiguration{
				Authorizers: []api.AuthorizerConfiguration{
					{},
				},
			},
			expectedErrList: field.ErrorList{field.Required(field.NewPath("type"), "")},
			knownTypes:      sets.NewString(string("Webhook")),
			repeatableTypes: sets.NewString(string("Webhook")),
		},
		{
			name: "authorizer names should be of non-zero length",
			configuration: api.AuthorizationConfiguration{
				Authorizers: []api.AuthorizerConfiguration{
					{
						Type: "Foo",
						Name: "",
					},
				},
			},
			expectedErrList: field.ErrorList{field.Required(field.NewPath("name"), "")},
			knownTypes:      sets.NewString(string("Foo")),
			repeatableTypes: sets.NewString(string("Webhook")),
		},
		{
			name: "authorizer names should be unique",
			configuration: api.AuthorizationConfiguration{
				Authorizers: []api.AuthorizerConfiguration{
					{
						Type: "Foo",
						Name: "foo",
					},
					{
						Type: "Bar",
						Name: "foo",
					},
				},
			},
			expectedErrList: field.ErrorList{field.Duplicate(field.NewPath("name"), "foo")},
			knownTypes:      sets.NewString(string("Foo"), string("Bar")),
			repeatableTypes: sets.NewString(string("Webhook")),
		},
		{
			name: "authorizer names should be DNS1123 labels",
			configuration: api.AuthorizationConfiguration{
				Authorizers: []api.AuthorizerConfiguration{
					{
						Type: "Foo",
						Name: "myauthorizer",
					},
				},
			},
			expectedErrList: field.ErrorList{},
			knownTypes:      sets.NewString(string("Foo")),
			repeatableTypes: sets.NewString(string("Webhook")),
		},
		{
			name: "authorizer names should be DNS1123 subdomains",
			configuration: api.AuthorizationConfiguration{
				Authorizers: []api.AuthorizerConfiguration{
					{
						Type: "Foo",
						Name: "foo.example.domain",
					},
				},
			},
			expectedErrList: field.ErrorList{},
			knownTypes:      sets.NewString(string("Foo")),
			repeatableTypes: sets.NewString(string("Webhook")),
		},
		{
			name: "authorizer names should not be invalid DNS1123 labels or subdomains",
			configuration: api.AuthorizationConfiguration{
				Authorizers: []api.AuthorizerConfiguration{
					{
						Type: "Foo",
						Name: "FOO.example.domain",
					},
				},
			},
			expectedErrList: field.ErrorList{field.Invalid(field.NewPath("name"), "FOO.example.domain", "")},
			knownTypes:      sets.NewString(string("Foo")),
			repeatableTypes: sets.NewString(string("Webhook")),
		},
		{
			name: "bare minimum configuration with Webhook",
			configuration: api.AuthorizationConfiguration{
				Authorizers: []api.AuthorizerConfiguration{
					{
						Type: "Webhook",
						Name: "default",
						Webhook: &api.WebhookConfiguration{
							Timeout:                                  metav1.Duration{Duration: 5 * time.Second},
							AuthorizedTTL:                            metav1.Duration{Duration: 5 * time.Minute},
							UnauthorizedTTL:                          metav1.Duration{Duration: 30 * time.Second},
							FailurePolicy:                            "NoOpinion",
							SubjectAccessReviewVersion:               "v1",
							MatchConditionSubjectAccessReviewVersion: "v1",
							ConnectionInfo: api.WebhookConnectionInfo{
								Type: "InClusterConfig",
							},
						},
					},
				},
			},
			expectedErrList: field.ErrorList{},
			knownTypes:      sets.NewString(string("Webhook")),
			repeatableTypes: sets.NewString(string("Webhook")),
		},
		{
			name: "bare minimum configuration with Webhook and MatchConditions",
			configuration: api.AuthorizationConfiguration{
				Authorizers: []api.AuthorizerConfiguration{
					{
						Type: "Webhook",
						Name: "default",
						Webhook: &api.WebhookConfiguration{
							Timeout:                                  metav1.Duration{Duration: 5 * time.Second},
							AuthorizedTTL:                            metav1.Duration{Duration: 5 * time.Minute},
							UnauthorizedTTL:                          metav1.Duration{Duration: 30 * time.Second},
							FailurePolicy:                            "NoOpinion",
							SubjectAccessReviewVersion:               "v1",
							MatchConditionSubjectAccessReviewVersion: "v1",
							ConnectionInfo: api.WebhookConnectionInfo{
								Type: "InClusterConfig",
							},
							MatchConditions: []api.WebhookMatchCondition{
								{
									Expression: "has(request.resourceAttributes) && request.resourceAttributes.namespace == 'kube-system'",
								},
								{
									Expression: "request.user == 'admin'",
								},
							},
						},
					},
				},
			},
			expectedErrList: field.ErrorList{},
			knownTypes:      sets.NewString(string("Webhook")),
			repeatableTypes: sets.NewString(string("Webhook")),
		},
		{
			name: "bare minimum configuration with multiple webhooks",
			configuration: api.AuthorizationConfiguration{
				Authorizers: []api.AuthorizerConfiguration{
					{
						Type: "Webhook",
						Name: "default",
						Webhook: &api.WebhookConfiguration{
							Timeout:                                  metav1.Duration{Duration: 5 * time.Second},
							AuthorizedTTL:                            metav1.Duration{Duration: 5 * time.Minute},
							UnauthorizedTTL:                          metav1.Duration{Duration: 30 * time.Second},
							FailurePolicy:                            "NoOpinion",
							SubjectAccessReviewVersion:               "v1",
							MatchConditionSubjectAccessReviewVersion: "v1",
							ConnectionInfo: api.WebhookConnectionInfo{
								Type: "InClusterConfig",
							},
						},
					},
					{
						Type: "Webhook",
						Name: "second-webhook",
						Webhook: &api.WebhookConfiguration{
							Timeout:                                  metav1.Duration{Duration: 5 * time.Second},
							AuthorizedTTL:                            metav1.Duration{Duration: 5 * time.Minute},
							UnauthorizedTTL:                          metav1.Duration{Duration: 30 * time.Second},
							FailurePolicy:                            "NoOpinion",
							SubjectAccessReviewVersion:               "v1",
							MatchConditionSubjectAccessReviewVersion: "v1",
							ConnectionInfo: api.WebhookConnectionInfo{
								Type: "InClusterConfig",
							},
						},
					},
				},
			},
			expectedErrList: field.ErrorList{},
			knownTypes:      sets.NewString(string("Webhook")),
			repeatableTypes: sets.NewString(string("Webhook")),
		},
		{
			name: "configuration with unknown types",
			configuration: api.AuthorizationConfiguration{
				Authorizers: []api.AuthorizerConfiguration{
					{
						Type: "Foo",
					},
				},
			},
			expectedErrList: field.ErrorList{field.NotSupported(field.NewPath("type"), "Foo", []string{"..."})},
			knownTypes:      sets.NewString(string("Webhook")),
			repeatableTypes: sets.NewString(string("Webhook")),
		},
		{
			name: "configuration with not repeatable types",
			configuration: api.AuthorizationConfiguration{
				Authorizers: []api.AuthorizerConfiguration{
					{
						Type: "Foo",
						Name: "foo-1",
					},
					{
						Type: "Foo",
						Name: "foo-2",
					},
				},
			},
			expectedErrList: field.ErrorList{field.Duplicate(field.NewPath("type"), "Foo")},
			knownTypes:      sets.NewString(string("Foo")),
			repeatableTypes: sets.NewString(string("Webhook")),
		},
		{
			name: "when type=Webhook, webhook needs to be defined",
			configuration: api.AuthorizationConfiguration{
				Authorizers: []api.AuthorizerConfiguration{
					{
						Type: "Webhook",
						Name: "default",
					},
				},
			},
			expectedErrList: field.ErrorList{field.Required(field.NewPath("webhook"), "required when type=Webhook")},
			knownTypes:      sets.NewString(string("Webhook")),
			repeatableTypes: sets.NewString(string("Webhook")),
		},
		{
			name: "when type!=Webhook, webhooks needs to be nil",
			configuration: api.AuthorizationConfiguration{
				Authorizers: []api.AuthorizerConfiguration{
					{
						Type:    "Foo",
						Name:    "foo",
						Webhook: &api.WebhookConfiguration{},
					},
				},
			},
			expectedErrList: field.ErrorList{field.Invalid(field.NewPath("webhook"), "non-null", "may only be specified when type=Webhook")},
			knownTypes:      sets.NewString(string("Foo")),
			repeatableTypes: sets.NewString(string("Webhook")),
		},
		{
			name: "timeout should be specified",
			configuration: api.AuthorizationConfiguration{
				Authorizers: []api.AuthorizerConfiguration{
					{
						Type: "Webhook",
						Name: "default",
						Webhook: &api.WebhookConfiguration{
							FailurePolicy:                            "NoOpinion",
							AuthorizedTTL:                            metav1.Duration{Duration: 5 * time.Minute},
							UnauthorizedTTL:                          metav1.Duration{Duration: 30 * time.Second},
							SubjectAccessReviewVersion:               "v1",
							MatchConditionSubjectAccessReviewVersion: "v1",
							ConnectionInfo: api.WebhookConnectionInfo{
								Type: "InClusterConfig",
							},
						},
					},
				},
			},
			expectedErrList: field.ErrorList{field.Required(field.NewPath("timeout"), "")},
			knownTypes:      sets.NewString(string("Webhook")),
			repeatableTypes: sets.NewString(string("Webhook")),
		},
		//
		{
			name: "timeout shouldn't be zero",
			configuration: api.AuthorizationConfiguration{
				Authorizers: []api.AuthorizerConfiguration{
					{
						Type: "Webhook",
						Name: "default",
						Webhook: &api.WebhookConfiguration{
							FailurePolicy:                            "NoOpinion",
							Timeout:                                  metav1.Duration{Duration: 0 * time.Second},
							AuthorizedTTL:                            metav1.Duration{Duration: 5 * time.Minute},
							UnauthorizedTTL:                          metav1.Duration{Duration: 30 * time.Second},
							SubjectAccessReviewVersion:               "v1",
							MatchConditionSubjectAccessReviewVersion: "v1",
							ConnectionInfo: api.WebhookConnectionInfo{
								Type: "InClusterConfig",
							},
						},
					},
				},
			},
			expectedErrList: field.ErrorList{field.Required(field.NewPath("timeout"), "")},
			knownTypes:      sets.NewString(string("Webhook")),
			repeatableTypes: sets.NewString(string("Webhook")),
		},
		{
			name: "timeout shouldn't be negative",
			configuration: api.AuthorizationConfiguration{
				Authorizers: []api.AuthorizerConfiguration{
					{
						Type: "Webhook",
						Name: "default",
						Webhook: &api.WebhookConfiguration{
							FailurePolicy:                            "NoOpinion",
							Timeout:                                  metav1.Duration{Duration: -30 * time.Second},
							AuthorizedTTL:                            metav1.Duration{Duration: 5 * time.Minute},
							UnauthorizedTTL:                          metav1.Duration{Duration: 30 * time.Second},
							SubjectAccessReviewVersion:               "v1",
							MatchConditionSubjectAccessReviewVersion: "v1",
							ConnectionInfo: api.WebhookConnectionInfo{
								Type: "InClusterConfig",
							},
						},
					},
				},
			},
			expectedErrList: field.ErrorList{field.Invalid(field.NewPath("timeout"), time.Duration(-30*time.Second).String(), "must be > 0s and <= 30s")},
			knownTypes:      sets.NewString(string("Webhook")),
			repeatableTypes: sets.NewString(string("Webhook")),
		},
		{
			name: "timeout shouldn't be greater than 30seconds",
			configuration: api.AuthorizationConfiguration{
				Authorizers: []api.AuthorizerConfiguration{
					{
						Type: "Webhook",
						Name: "default",
						Webhook: &api.WebhookConfiguration{
							FailurePolicy:                            "NoOpinion",
							Timeout:                                  metav1.Duration{Duration: 60 * time.Second},
							AuthorizedTTL:                            metav1.Duration{Duration: 5 * time.Minute},
							UnauthorizedTTL:                          metav1.Duration{Duration: 30 * time.Second},
							SubjectAccessReviewVersion:               "v1",
							MatchConditionSubjectAccessReviewVersion: "v1",
							ConnectionInfo: api.WebhookConnectionInfo{
								Type: "InClusterConfig",
							},
						},
					},
				},
			},
			expectedErrList: field.ErrorList{field.Invalid(field.NewPath("timeout"), time.Duration(60*time.Second).String(), "must be > 0s and <= 30s")},
			knownTypes:      sets.NewString(string("Webhook")),
			repeatableTypes: sets.NewString(string("Webhook")),
		},
		{
			name: "authorizedTTL should be defined ",
			configuration: api.AuthorizationConfiguration{
				Authorizers: []api.AuthorizerConfiguration{
					{
						Type: "Webhook",
						Name: "default",
						Webhook: &api.WebhookConfiguration{
							FailurePolicy:                            "NoOpinion",
							Timeout:                                  metav1.Duration{Duration: 5 * time.Second},
							UnauthorizedTTL:                          metav1.Duration{Duration: 30 * time.Second},
							SubjectAccessReviewVersion:               "v1",
							MatchConditionSubjectAccessReviewVersion: "v1",
							ConnectionInfo: api.WebhookConnectionInfo{
								Type: "InClusterConfig",
							},
						},
					},
				},
			},
			expectedErrList: field.ErrorList{field.Required(field.NewPath("authorizedTTL"), "")},
			knownTypes:      sets.NewString(string("Webhook")),
			repeatableTypes: sets.NewString(string("Webhook")),
		},
		{
			name: "authorizedTTL shouldn't be negative",
			configuration: api.AuthorizationConfiguration{
				Authorizers: []api.AuthorizerConfiguration{
					{
						Type: "Webhook",
						Name: "default",
						Webhook: &api.WebhookConfiguration{
							FailurePolicy:                            "NoOpinion",
							Timeout:                                  metav1.Duration{Duration: 5 * time.Second},
							AuthorizedTTL:                            metav1.Duration{Duration: -30 * time.Second},
							UnauthorizedTTL:                          metav1.Duration{Duration: 30 * time.Second},
							SubjectAccessReviewVersion:               "v1",
							MatchConditionSubjectAccessReviewVersion: "v1",
							ConnectionInfo: api.WebhookConnectionInfo{
								Type: "InClusterConfig",
							},
						},
					},
				},
			},
			expectedErrList: field.ErrorList{field.Invalid(field.NewPath("authorizedTTL"), time.Duration(-30*time.Second).String(), "must be > 0s")},
			knownTypes:      sets.NewString(string("Webhook")),
			repeatableTypes: sets.NewString(string("Webhook")),
		},
		{
			name: "unauthorizedTTL should be defined ",
			configuration: api.AuthorizationConfiguration{
				Authorizers: []api.AuthorizerConfiguration{
					{
						Type: "Webhook",
						Name: "default",
						Webhook: &api.WebhookConfiguration{
							FailurePolicy:                            "NoOpinion",
							Timeout:                                  metav1.Duration{Duration: 5 * time.Second},
							AuthorizedTTL:                            metav1.Duration{Duration: 5 * time.Minute},
							SubjectAccessReviewVersion:               "v1",
							MatchConditionSubjectAccessReviewVersion: "v1",
							ConnectionInfo: api.WebhookConnectionInfo{
								Type: "InClusterConfig",
							},
						},
					},
				},
			},
			expectedErrList: field.ErrorList{field.Required(field.NewPath("unauthorizedTTL"), "")},
			knownTypes:      sets.NewString(string("Webhook")),
			repeatableTypes: sets.NewString(string("Webhook")),
		},
		{
			name: "unauthorizedTTL shouldn't be negative",
			configuration: api.AuthorizationConfiguration{
				Authorizers: []api.AuthorizerConfiguration{
					{
						Type: "Webhook",
						Name: "default",
						Webhook: &api.WebhookConfiguration{
							FailurePolicy:                            "NoOpinion",
							Timeout:                                  metav1.Duration{Duration: 5 * time.Second},
							AuthorizedTTL:                            metav1.Duration{Duration: 5 * time.Minute},
							UnauthorizedTTL:                          metav1.Duration{Duration: -30 * time.Second},
							SubjectAccessReviewVersion:               "v1",
							MatchConditionSubjectAccessReviewVersion: "v1",
							ConnectionInfo: api.WebhookConnectionInfo{
								Type: "InClusterConfig",
							},
						},
					},
				},
			},
			expectedErrList: field.ErrorList{field.Invalid(field.NewPath("unauthorizedTTL"), time.Duration(-30*time.Second).String(), "must be > 0s")},
			knownTypes:      sets.NewString(string("Webhook")),
			repeatableTypes: sets.NewString(string("Webhook")),
		},
		{
			name: "SAR should be defined",
			configuration: api.AuthorizationConfiguration{
				Authorizers: []api.AuthorizerConfiguration{
					{
						Type: "Webhook",
						Name: "default",
						Webhook: &api.WebhookConfiguration{
							Timeout:                                  metav1.Duration{Duration: 5 * time.Second},
							AuthorizedTTL:                            metav1.Duration{Duration: 5 * time.Minute},
							UnauthorizedTTL:                          metav1.Duration{Duration: 30 * time.Second},
							MatchConditionSubjectAccessReviewVersion: "v1",
							FailurePolicy:                            "NoOpinion",
							ConnectionInfo: api.WebhookConnectionInfo{
								Type: "InClusterConfig",
							},
						},
					},
				},
			},
			expectedErrList: field.ErrorList{field.Required(field.NewPath("subjectAccessReviewVersion"), "")},
			knownTypes:      sets.NewString(string("Webhook")),
			repeatableTypes: sets.NewString(string("Webhook")),
		},
		{
			name: "SAR should be one of v1 and v1beta1",
			configuration: api.AuthorizationConfiguration{
				Authorizers: []api.AuthorizerConfiguration{
					{
						Type: "Webhook",
						Name: "default",
						Webhook: &api.WebhookConfiguration{
							Timeout:                                  metav1.Duration{Duration: 5 * time.Second},
							AuthorizedTTL:                            metav1.Duration{Duration: 5 * time.Minute},
							UnauthorizedTTL:                          metav1.Duration{Duration: 30 * time.Second},
							FailurePolicy:                            "NoOpinion",
							SubjectAccessReviewVersion:               "v2beta1",
							MatchConditionSubjectAccessReviewVersion: "v1",
							ConnectionInfo: api.WebhookConnectionInfo{
								Type: "InClusterConfig",
							},
						},
					},
				},
			},
			expectedErrList: field.ErrorList{field.NotSupported(field.NewPath("subjectAccessReviewVersion"), "v2beta1", []string{"v1", "v1beta1"})},
			knownTypes:      sets.NewString(string("Webhook")),
			repeatableTypes: sets.NewString(string("Webhook")),
		},
		{
			name: "MatchConditionSAR should be defined",
			configuration: api.AuthorizationConfiguration{
				Authorizers: []api.AuthorizerConfiguration{
					{
						Type: "Webhook",
						Name: "default",
						Webhook: &api.WebhookConfiguration{
							Timeout:                    metav1.Duration{Duration: 5 * time.Second},
							AuthorizedTTL:              metav1.Duration{Duration: 5 * time.Minute},
							UnauthorizedTTL:            metav1.Duration{Duration: 30 * time.Second},
							FailurePolicy:              "NoOpinion",
							SubjectAccessReviewVersion: "v1",
							ConnectionInfo: api.WebhookConnectionInfo{
								Type: "InClusterConfig",
							},
							MatchConditions: []api.WebhookMatchCondition{{Expression: "true"}},
						},
					},
				},
			},
			expectedErrList: field.ErrorList{field.Required(field.NewPath("matchConditionSubjectAccessReviewVersion"), "")},
			knownTypes:      sets.NewString(string("Webhook")),
			repeatableTypes: sets.NewString(string("Webhook")),
		},
		{
			name: "MatchConditionSAR must not be anything other than v1",
			configuration: api.AuthorizationConfiguration{
				Authorizers: []api.AuthorizerConfiguration{
					{
						Type: "Webhook",
						Name: "default",
						Webhook: &api.WebhookConfiguration{
							Timeout:                                  metav1.Duration{Duration: 5 * time.Second},
							AuthorizedTTL:                            metav1.Duration{Duration: 5 * time.Minute},
							UnauthorizedTTL:                          metav1.Duration{Duration: 30 * time.Second},
							FailurePolicy:                            "NoOpinion",
							SubjectAccessReviewVersion:               "v1",
							MatchConditionSubjectAccessReviewVersion: "v1beta1",
							ConnectionInfo: api.WebhookConnectionInfo{
								Type: "InClusterConfig",
							},
						},
					},
				},
			},
			expectedErrList: field.ErrorList{field.NotSupported(field.NewPath("matchConditionSubjectAccessReviewVersion"), "v1beta1", []string{"v1"})},
			knownTypes:      sets.NewString(string("Webhook")),
			repeatableTypes: sets.NewString(string("Webhook")),
		},
		{
			name: "failurePolicy should be defined",
			configuration: api.AuthorizationConfiguration{
				Authorizers: []api.AuthorizerConfiguration{
					{
						Type: "Webhook",
						Name: "default",
						Webhook: &api.WebhookConfiguration{
							Timeout:                                  metav1.Duration{Duration: 5 * time.Second},
							AuthorizedTTL:                            metav1.Duration{Duration: 5 * time.Minute},
							UnauthorizedTTL:                          metav1.Duration{Duration: 30 * time.Second},
							SubjectAccessReviewVersion:               "v1",
							MatchConditionSubjectAccessReviewVersion: "v1",
							ConnectionInfo: api.WebhookConnectionInfo{
								Type: "InClusterConfig",
							},
						},
					},
				},
			},
			expectedErrList: field.ErrorList{field.Required(field.NewPath("failurePolicy"), "")},
			knownTypes:      sets.NewString(string("Webhook")),
			repeatableTypes: sets.NewString(string("Webhook")),
		},
		{
			name: "failurePolicy should be one of \"NoOpinion\" or \"Deny\"",
			configuration: api.AuthorizationConfiguration{
				Authorizers: []api.AuthorizerConfiguration{
					{
						Type: "Webhook",
						Name: "default",
						Webhook: &api.WebhookConfiguration{
							Timeout:                                  metav1.Duration{Duration: 5 * time.Second},
							AuthorizedTTL:                            metav1.Duration{Duration: 5 * time.Minute},
							UnauthorizedTTL:                          metav1.Duration{Duration: 30 * time.Second},
							FailurePolicy:                            "AlwaysAllow",
							SubjectAccessReviewVersion:               "v1",
							MatchConditionSubjectAccessReviewVersion: "v1",
							ConnectionInfo: api.WebhookConnectionInfo{
								Type: "InClusterConfig",
							},
						},
					},
				},
			},
			expectedErrList: field.ErrorList{field.NotSupported(field.NewPath("failurePolicy"), "AlwaysAllow", []string{"NoOpinion", "Deny"})},
			knownTypes:      sets.NewString(string("Webhook")),
			repeatableTypes: sets.NewString(string("Webhook")),
		},
		{
			name: "connectionInfo should be defined",
			configuration: api.AuthorizationConfiguration{
				Authorizers: []api.AuthorizerConfiguration{
					{
						Type: "Webhook",
						Name: "default",
						Webhook: &api.WebhookConfiguration{
							Timeout:                                  metav1.Duration{Duration: 5 * time.Second},
							AuthorizedTTL:                            metav1.Duration{Duration: 5 * time.Minute},
							UnauthorizedTTL:                          metav1.Duration{Duration: 30 * time.Second},
							FailurePolicy:                            "NoOpinion",
							SubjectAccessReviewVersion:               "v1",
							MatchConditionSubjectAccessReviewVersion: "v1",
						},
					},
				},
			},
			expectedErrList: field.ErrorList{field.Required(field.NewPath("connectionInfo"), "")},
			knownTypes:      sets.NewString(string("Webhook")),
			repeatableTypes: sets.NewString(string("Webhook")),
		},
		{
			name: "connectionInfo should be one of InClusterConfig or KubeConfigFile",
			configuration: api.AuthorizationConfiguration{
				Authorizers: []api.AuthorizerConfiguration{
					{
						Type: "Webhook",
						Name: "default",
						Webhook: &api.WebhookConfiguration{
							Timeout:                                  metav1.Duration{Duration: 5 * time.Second},
							AuthorizedTTL:                            metav1.Duration{Duration: 5 * time.Minute},
							UnauthorizedTTL:                          metav1.Duration{Duration: 30 * time.Second},
							FailurePolicy:                            "NoOpinion",
							SubjectAccessReviewVersion:               "v1",
							MatchConditionSubjectAccessReviewVersion: "v1",
							ConnectionInfo: api.WebhookConnectionInfo{
								Type: "ExternalClusterConfig",
							},
						},
					},
				},
			},
			expectedErrList: field.ErrorList{
				field.NotSupported(field.NewPath("connectionInfo"), api.WebhookConnectionInfo{Type: "ExternalClusterConfig"}, []string{"InClusterConfig", "KubeConfigFile"}),
			},
			knownTypes:      sets.NewString(string("Webhook")),
			repeatableTypes: sets.NewString(string("Webhook")),
		},
		{
			name: "if connectionInfo=InClusterConfig, then kubeConfigFile should be nil",
			configuration: api.AuthorizationConfiguration{
				Authorizers: []api.AuthorizerConfiguration{
					{
						Type: "Webhook",
						Name: "default",
						Webhook: &api.WebhookConfiguration{
							Timeout:                                  metav1.Duration{Duration: 5 * time.Second},
							AuthorizedTTL:                            metav1.Duration{Duration: 5 * time.Minute},
							UnauthorizedTTL:                          metav1.Duration{Duration: 30 * time.Second},
							FailurePolicy:                            "NoOpinion",
							SubjectAccessReviewVersion:               "v1",
							MatchConditionSubjectAccessReviewVersion: "v1",
							ConnectionInfo: api.WebhookConnectionInfo{
								Type:           "InClusterConfig",
								KubeConfigFile: new(string),
							},
						},
					},
				},
			},
			expectedErrList: field.ErrorList{
				field.Invalid(field.NewPath("connectionInfo", "kubeConfigFile"), "", "can only be set when type=KubeConfigFile"),
			},
			knownTypes:      sets.NewString(string("Webhook")),
			repeatableTypes: sets.NewString(string("Webhook")),
		},
		{
			name: "if connectionInfo=KubeConfigFile, then KubeConfigFile should be defined",
			configuration: api.AuthorizationConfiguration{
				Authorizers: []api.AuthorizerConfiguration{
					{
						Type: "Webhook",
						Name: "default",
						Webhook: &api.WebhookConfiguration{
							Timeout:                                  metav1.Duration{Duration: 5 * time.Second},
							AuthorizedTTL:                            metav1.Duration{Duration: 5 * time.Minute},
							UnauthorizedTTL:                          metav1.Duration{Duration: 30 * time.Second},
							FailurePolicy:                            "NoOpinion",
							SubjectAccessReviewVersion:               "v1",
							MatchConditionSubjectAccessReviewVersion: "v1",
							ConnectionInfo: api.WebhookConnectionInfo{
								Type: "KubeConfigFile",
							},
						},
					},
				},
			},
			expectedErrList: field.ErrorList{field.Required(field.NewPath("kubeConfigFile"), "")},
			knownTypes:      sets.NewString(string("Webhook")),
			repeatableTypes: sets.NewString(string("Webhook")),
		},
		{
			name: "if connectionInfo=KubeConfigFile, then KubeConfigFile should be defined, must be an absolute path, should exist, shouldn't be a symlink",
			configuration: api.AuthorizationConfiguration{
				Authorizers: []api.AuthorizerConfiguration{
					{
						Type: "Webhook",
						Name: "default",
						Webhook: &api.WebhookConfiguration{
							Timeout:                                  metav1.Duration{Duration: 5 * time.Second},
							AuthorizedTTL:                            metav1.Duration{Duration: 5 * time.Minute},
							UnauthorizedTTL:                          metav1.Duration{Duration: 30 * time.Second},
							FailurePolicy:                            "NoOpinion",
							SubjectAccessReviewVersion:               "v1",
							MatchConditionSubjectAccessReviewVersion: "v1",
							ConnectionInfo: api.WebhookConnectionInfo{
								Type:           "KubeConfigFile",
								KubeConfigFile: &badKubeConfigFile,
							},
						},
					},
				},
			},
			expectedErrList: field.ErrorList{field.Invalid(field.NewPath("kubeConfigFile"), badKubeConfigFile, "must be an absolute path")},
			knownTypes:      sets.NewString(string("Webhook")),
			repeatableTypes: sets.NewString(string("Webhook")),
		},
		{
			name: "if connectionInfo=KubeConfigFile, an existent file needs to be passed",
			configuration: api.AuthorizationConfiguration{
				Authorizers: []api.AuthorizerConfiguration{
					{
						Type: "Webhook",
						Name: "default",
						Webhook: &api.WebhookConfiguration{
							Timeout:                                  metav1.Duration{Duration: 5 * time.Second},
							AuthorizedTTL:                            metav1.Duration{Duration: 5 * time.Minute},
							UnauthorizedTTL:                          metav1.Duration{Duration: 30 * time.Second},
							FailurePolicy:                            "NoOpinion",
							SubjectAccessReviewVersion:               "v1",
							MatchConditionSubjectAccessReviewVersion: "v1",
							ConnectionInfo: api.WebhookConnectionInfo{
								Type:           "KubeConfigFile",
								KubeConfigFile: &tempKubeConfigFilePath,
							},
						},
					},
				},
			},
			expectedErrList: field.ErrorList{},
			knownTypes:      sets.NewString(string("Webhook")),
			repeatableTypes: sets.NewString(string("Webhook")),
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			errList := ValidateAuthorizationConfiguration(nil, &test.configuration, test.knownTypes, test.repeatableTypes)
			if len(errList) != len(test.expectedErrList) {
				t.Errorf("expected %d errs, got %d, errors %v", len(test.expectedErrList), len(errList), errList)
			}
			if len(errList) == len(test.expectedErrList) {
				for i, expected := range test.expectedErrList {
					if expected.Type.String() != errList[i].Type.String() {
						t.Errorf("expected err type %s, got %s",
							expected.Type.String(),
							errList[i].Type.String())
					}
					if expected.BadValue != errList[i].BadValue {
						t.Errorf("expected bad value '%s', got '%s'",
							expected.BadValue,
							errList[i].BadValue)
					}
				}
			}
		})

	}
}

func TestValidateAndCompileMatchConditions(t *testing.T) {
	testCases := []struct {
		name            string
		matchConditions []api.WebhookMatchCondition
		featureEnabled  bool
		expectedErr     string
	}{
		{
			name: "match conditions are used With feature enabled",
			matchConditions: []api.WebhookMatchCondition{
				{
					Expression: "has(request.resourceAttributes) && request.resourceAttributes.namespace == 'kube-system'",
				},
				{
					Expression: "request.user == 'admin'",
				},
			},
			featureEnabled: true,
			expectedErr:    "",
		},
		{
			name: "should fail when match conditions are used without feature enabled",
			matchConditions: []api.WebhookMatchCondition{
				{
					Expression: "has(request.resourceAttributes) && request.resourceAttributes.namespace == 'kube-system'",
				},
				{
					Expression: "request.user == 'admin'",
				},
			},
			featureEnabled: false,
			expectedErr:    `matchConditions: Invalid value: "": matchConditions are not supported when StructuredAuthorizationConfiguration feature gate is disabled`,
		},
		{
			name:            "no matchConditions should not require feature enablement",
			matchConditions: []api.WebhookMatchCondition{},
			featureEnabled:  false,
			expectedErr:     "",
		},
		{
			name: "match conditions with invalid expressions",
			matchConditions: []api.WebhookMatchCondition{
				{
					Expression: "  ",
				},
			},
			featureEnabled: true,
			expectedErr:    "matchConditions[0].expression: Required value",
		},
		{
			name: "match conditions with duplicate expressions",
			matchConditions: []api.WebhookMatchCondition{
				{
					Expression: "request.user == 'admin'",
				},
				{
					Expression: "request.user == 'admin'",
				},
			},
			featureEnabled: true,
			expectedErr:    `matchConditions[1].expression: Duplicate value: "request.user == 'admin'"`,
		},
		{
			name: "match conditions with undeclared reference",
			matchConditions: []api.WebhookMatchCondition{
				{
					Expression: "test",
				},
			},
			featureEnabled: true,
			expectedErr:    "matchConditions[0].expression: Invalid value: \"test\": compilation failed: ERROR: <input>:1:1: undeclared reference to 'test' (in container '')\n | test\n | ^",
		},
		{
			name: "match conditions with bad return type",
			matchConditions: []api.WebhookMatchCondition{
				{
					Expression: "request.user = 'test'",
				},
			},
			featureEnabled: true,
			expectedErr:    "matchConditions[0].expression: Invalid value: \"request.user = 'test'\": compilation failed: ERROR: <input>:1:14: Syntax error: token recognition error at: '= '\n | request.user = 'test'\n | .............^\nERROR: <input>:1:16: Syntax error: extraneous input ''test'' expecting <EOF>\n | request.user = 'test'\n | ...............^",
		},
	}

	for _, tt := range testCases {
		t.Run(tt.name, func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.StructuredAuthorizationConfiguration, tt.featureEnabled)()
			celMatcher, errList := ValidateAndCompileMatchConditions(tt.matchConditions)
			if len(tt.expectedErr) == 0 && len(tt.matchConditions) > 0 && len(errList) == 0 && celMatcher == nil {
				t.Errorf("celMatcher should not be nil when there are matchCondition and no error returned")
			}
			got := errList.ToAggregate()
			if d := cmp.Diff(tt.expectedErr, errString(got)); d != "" {
				t.Fatalf("ValidateAndCompileMatchConditions validation mismatch (-want +got):\n%s", d)
			}
		})
	}
}
