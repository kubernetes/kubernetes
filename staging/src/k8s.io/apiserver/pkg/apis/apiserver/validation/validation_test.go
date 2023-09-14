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
	"os"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/sets"
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
			name: "type is required if an authorizer is defined",
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
			name: "bare minimum configuration with Webhook",
			configuration: api.AuthorizationConfiguration{
				Authorizers: []api.AuthorizerConfiguration{
					{
						Type: "Webhook",
						Webhook: &api.WebhookConfiguration{
							Name:                                     "default",
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
			name: "bare minimum configuration with multiple webhooks",
			configuration: api.AuthorizationConfiguration{
				Authorizers: []api.AuthorizerConfiguration{
					{
						Type: "Webhook",
						Webhook: &api.WebhookConfiguration{
							Name:                                     "default",
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
						Webhook: &api.WebhookConfiguration{
							Name:                                     "second-webhook",
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
					},
					{
						Type: "Foo",
					},
				},
			},
			expectedErrList: field.ErrorList{field.Duplicate(field.NewPath("type"), "Foo")},
			knownTypes:      sets.NewString([]string{string("Foo"), string("Webhook")}...),
			repeatableTypes: sets.NewString(string("Webhook")),
		},
		{
			name: "when type=Webhook, webhook needs to be defined",
			configuration: api.AuthorizationConfiguration{
				Authorizers: []api.AuthorizerConfiguration{
					{
						Type: "Webhook",
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
						Webhook: &api.WebhookConfiguration{},
					},
				},
			},
			expectedErrList: field.ErrorList{field.Invalid(field.NewPath("webhook"), "non-null", "may only be specified when type=Webhook")},
			knownTypes:      sets.NewString(string("Foo")),
			repeatableTypes: sets.NewString(string("Webhook")),
		},
		{
			name: "webhook name should be of non-zero length",
			configuration: api.AuthorizationConfiguration{
				Authorizers: []api.AuthorizerConfiguration{
					{
						Type: "Webhook",
						Webhook: &api.WebhookConfiguration{
							Name:                                     "",
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
			expectedErrList: field.ErrorList{field.Required(field.NewPath("name"), "")},
			knownTypes:      sets.NewString(string("Webhook")),
			repeatableTypes: sets.NewString(string("Webhook")),
		},
		{
			name: "webhook names should be unique",
			configuration: api.AuthorizationConfiguration{
				Authorizers: []api.AuthorizerConfiguration{
					{
						Type: "Webhook",
						Webhook: &api.WebhookConfiguration{
							Name:                                     "name-1",
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
						Webhook: &api.WebhookConfiguration{
							Name:                                     "name-1",
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
			expectedErrList: field.ErrorList{field.Duplicate(field.NewPath("name"), "name-1")},
			knownTypes:      sets.NewString(string("Webhook")),
			repeatableTypes: sets.NewString(string("Webhook")),
		},
		{
			name: "webhook names should be DNS1123 labels",
			configuration: api.AuthorizationConfiguration{
				Authorizers: []api.AuthorizerConfiguration{
					{
						Type: "Webhook",
						Webhook: &api.WebhookConfiguration{
							Name:                                     "mywebhookname",
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
			name: "webhook names should be DNS1123 subdomains",
			configuration: api.AuthorizationConfiguration{
				Authorizers: []api.AuthorizerConfiguration{
					{
						Type: "Webhook",
						Webhook: &api.WebhookConfiguration{
							Name:                                     "webhookname.example.domain",
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
			name: "webhook names should not be invalid DNS1123 labels or subdomains",
			configuration: api.AuthorizationConfiguration{
				Authorizers: []api.AuthorizerConfiguration{
					{
						Type: "Webhook",
						Webhook: &api.WebhookConfiguration{
							Name:                                     "WEBHOOKNAME.example.domain",
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
			expectedErrList: field.ErrorList{field.Invalid(field.NewPath("name"), "WEBHOOKNAME.example.domain", "")},
			knownTypes:      sets.NewString(string("Webhook")),
			repeatableTypes: sets.NewString(string("Webhook")),
		},
		{
			name: "timeout should be specified",
			configuration: api.AuthorizationConfiguration{
				Authorizers: []api.AuthorizerConfiguration{
					{
						Type: "Webhook",
						Webhook: &api.WebhookConfiguration{
							Name:                                     "default",
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
						Webhook: &api.WebhookConfiguration{
							Name:                                     "default",
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
						Webhook: &api.WebhookConfiguration{
							Name:                                     "default",
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
						Webhook: &api.WebhookConfiguration{
							Name:                                     "default",
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
						Webhook: &api.WebhookConfiguration{
							Name:                                     "default",
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
						Webhook: &api.WebhookConfiguration{
							Name:                                     "default",
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
						Webhook: &api.WebhookConfiguration{
							Name:                                     "default",
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
						Webhook: &api.WebhookConfiguration{
							Name:                                     "default",
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
						Webhook: &api.WebhookConfiguration{
							Name:                                     "default",
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
						Webhook: &api.WebhookConfiguration{
							Name:                                     "default",
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
						Webhook: &api.WebhookConfiguration{
							Name:                       "default",
							Timeout:                    metav1.Duration{Duration: 5 * time.Second},
							AuthorizedTTL:              metav1.Duration{Duration: 5 * time.Minute},
							UnauthorizedTTL:            metav1.Duration{Duration: 30 * time.Second},
							FailurePolicy:              "NoOpinion",
							SubjectAccessReviewVersion: "v1",
							ConnectionInfo: api.WebhookConnectionInfo{
								Type: "InClusterConfig",
							},
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
						Webhook: &api.WebhookConfiguration{
							Name:                                     "default",
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
						Webhook: &api.WebhookConfiguration{
							Name:                                     "default",
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
						Webhook: &api.WebhookConfiguration{
							Name:                                     "default",
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
						Webhook: &api.WebhookConfiguration{
							Name:                                     "default",
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
						Webhook: &api.WebhookConfiguration{
							Name:                                     "default",
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
						Webhook: &api.WebhookConfiguration{
							Name:                                     "default",
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
						Webhook: &api.WebhookConfiguration{
							Name:                                     "default",
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
						Webhook: &api.WebhookConfiguration{
							Name:                                     "default",
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
						Webhook: &api.WebhookConfiguration{
							Name:                                     "default",
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

		// TODO: When the CEL expression validator is implemented, add a few test cases to typecheck the expression
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			errList := ValidateAuthorizationConfiguration(nil, &test.configuration, test.knownTypes, test.repeatableTypes)
			if len(errList) != len(test.expectedErrList) {
				t.Errorf("expected %d errs, got %d, errors %v", len(test.expectedErrList), len(errList), errList)
			}

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
		})

	}
}
