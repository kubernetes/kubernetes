package oauth

import (
	"fmt"
	"reflect"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"

	configv1 "github.com/openshift/api/config/v1"
)

const wrongConfigMapSecretErrMsg string = "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')"

func htpasswdIDP() configv1.IdentityProviderConfig {
	return configv1.IdentityProviderConfig{
		Type: configv1.IdentityProviderTypeHTPasswd,
		HTPasswd: &configv1.HTPasswdIdentityProvider{
			FileData: configv1.SecretNameReference{
				Name: "innocent.llama",
			},
		},
	}
}

func TestValidateOAuthSpec(t *testing.T) {
	doubledIdPs := configv1.IdentityProviderConfig{
		Type: configv1.IdentityProviderTypeHTPasswd,
		HTPasswd: &configv1.HTPasswdIdentityProvider{
			FileData: configv1.SecretNameReference{
				Name: "innocent.llama",
			},
		},
		GitLab: &configv1.GitLabIdentityProvider{
			ClientID:     "masterOfInstances",
			ClientSecret: configv1.SecretNameReference{Name: "secret-gitlab-secret"},
			URL:          "https://thisgitlabinstancerighthere.com",
			CA:           configv1.ConfigMapNameReference{Name: "letsencrypt-for-gitlab.instance"},
		},
	}

	type args struct {
		spec configv1.OAuthSpec
	}
	tests := []struct {
		name string
		args args
		want field.ErrorList
	}{
		{
			name: "empty object",
			args: args{
				spec: configv1.OAuthSpec{},
			},
		},
		{
			name: "more than one challenge issuing IdPs",
			args: args{
				spec: configv1.OAuthSpec{
					IdentityProviders: []configv1.IdentityProvider{
						{
							Name:                   "htpasswd",
							IdentityProviderConfig: htpasswdIDP(),
						},
						{
							Name:                   "ldap",
							IdentityProviderConfig: ldapIDP(),
						},
					},
				},
			},
		},
		{
			name: "more than one challenge redirecting IdPs",
			args: args{
				spec: configv1.OAuthSpec{
					IdentityProviders: []configv1.IdentityProvider{
						{
							Name:                   "sso1",
							IdentityProviderConfig: requestHeaderIDP(true, true),
						},
						{
							Name:                   "sso2",
							IdentityProviderConfig: requestHeaderIDP(true, false),
						},
					},
				},
			},
			want: field.ErrorList{
				field.Invalid(field.NewPath("spec", "identityProviders"), "<omitted>", "only one identity provider can redirect clients requesting an authentication challenge, found: sso1, sso2"),
			},
		},
		{
			name: "mixing challenge issuing and redirecting IdPs",
			args: args{
				spec: configv1.OAuthSpec{
					IdentityProviders: []configv1.IdentityProvider{
						{
							Name:                   "sso",
							IdentityProviderConfig: requestHeaderIDP(true, false),
						},
						{
							Name:                   "ldap",
							IdentityProviderConfig: ldapIDP(),
						},
					},
				},
			},
			want: field.ErrorList{
				field.Invalid(field.NewPath("spec", "identityProviders"), "<omitted>", "cannot mix providers that redirect clients requesting auth challenges (sso) with providers issuing challenges to those clients (ldap)"),
			},
		},
		{
			name: "two IdPs with the same name",
			args: args{
				spec: configv1.OAuthSpec{
					IdentityProviders: []configv1.IdentityProvider{
						{
							Name:                   "aname",
							IdentityProviderConfig: htpasswdIDP(),
						},
						{
							Name:                   "bname",
							IdentityProviderConfig: htpasswdIDP(),
						},
						{
							Name:                   "aname",
							IdentityProviderConfig: htpasswdIDP(),
						},
						{
							Name:                   "cname",
							IdentityProviderConfig: htpasswdIDP(),
						},
					},
				},
			},
			want: field.ErrorList{
				field.Invalid(field.NewPath("spec", "identityProviders").Index(2).Child("name"), "aname", "must have a unique name"),
			},
		},
		{
			name: "negative token inactivity timeout",
			args: args{
				spec: configv1.OAuthSpec{
					TokenConfig: configv1.TokenConfig{
						AccessTokenInactivityTimeout: &metav1.Duration{Duration: -50 * time.Second},
					},
				},
			},
			want: field.ErrorList{
				field.Invalid(field.NewPath("spec", "tokenConfig", "accessTokenInactivityTimeout"), metav1.Duration{Duration: -50 * time.Second}, fmt.Sprintf("the minimum acceptable token timeout value is %d seconds", MinimumInactivityTimeoutSeconds)),
			},
		},
		{
			name: "positive token inactivity timeout",
			args: args{
				spec: configv1.OAuthSpec{
					TokenConfig: configv1.TokenConfig{
						AccessTokenInactivityTimeout: &metav1.Duration{Duration: 32578 * time.Second},
					},
				},
			},
		},
		{
			name: "zero token inactivity timeout",
			args: args{
				spec: configv1.OAuthSpec{
					TokenConfig: configv1.TokenConfig{
						AccessTokenInactivityTimeout: &metav1.Duration{Duration: 0},
					},
				},
			},
			want: field.ErrorList{
				field.Invalid(field.NewPath("spec", "tokenConfig", "accessTokenInactivityTimeout"), metav1.Duration{Duration: 0 * time.Second}, fmt.Sprintf("the minimum acceptable token timeout value is %d seconds", MinimumInactivityTimeoutSeconds)),
			},
		},
		{
			name: "token inactivity timeout lower than the api constant minimum",
			args: args{
				spec: configv1.OAuthSpec{
					TokenConfig: configv1.TokenConfig{
						AccessTokenInactivityTimeout: &metav1.Duration{Duration: 250 * time.Second},
					},
				},
			},
			want: field.ErrorList{
				field.Invalid(field.NewPath("spec", "tokenConfig", "accessTokenInactivityTimeout"), metav1.Duration{Duration: 250 * time.Second}, fmt.Sprintf("the minimum acceptable token timeout value is %d seconds", MinimumInactivityTimeoutSeconds)),
			},
		},
		{
			name: "negative token max age",
			args: args{
				spec: configv1.OAuthSpec{
					TokenConfig: configv1.TokenConfig{
						AccessTokenMaxAgeSeconds: -20,
					},
				},
			},
			want: field.ErrorList{
				field.Invalid(field.NewPath("spec", "tokenConfig", "accessTokenMaxAgeSeconds"), -20, "must be a positive integer or 0"),
			},
		},
		{
			name: "positive token max age",
			args: args{
				spec: configv1.OAuthSpec{
					TokenConfig: configv1.TokenConfig{
						AccessTokenMaxAgeSeconds: 213123,
					},
				},
			},
		},
		{
			name: "zero token max age",
			args: args{
				spec: configv1.OAuthSpec{
					TokenConfig: configv1.TokenConfig{
						AccessTokenMaxAgeSeconds: 0,
					},
				},
			},
		},
		{
			name: "template names all messed up",
			args: args{
				spec: configv1.OAuthSpec{
					Templates: configv1.OAuthTemplates{
						Login:             configv1.SecretNameReference{Name: "/this/is/wrong.html"},
						ProviderSelection: configv1.SecretNameReference{Name: "also_wrong"},
						Error:             configv1.SecretNameReference{Name: "the&very+woRst"},
					},
				},
			},
			want: field.ErrorList{
				field.Invalid(field.NewPath("spec", "templates", "login", "name"), "/this/is/wrong.html", wrongConfigMapSecretErrMsg),
				field.Invalid(field.NewPath("spec", "templates", "providerSelection", "name"), "also_wrong", wrongConfigMapSecretErrMsg),
				field.Invalid(field.NewPath("spec", "templates", "error", "name"), "the&very+woRst", wrongConfigMapSecretErrMsg),
			},
		},
		{
			name: "everything set properly",
			args: args{
				spec: configv1.OAuthSpec{
					IdentityProviders: []configv1.IdentityProvider{
						{
							Name:                   "some_httpasswd",
							IdentityProviderConfig: htpasswdIDP(),
						},
						{
							Name:                   "sso",
							IdentityProviderConfig: requestHeaderIDP(false, true),
						},
					},
					TokenConfig: configv1.TokenConfig{
						AccessTokenInactivityTimeout: &metav1.Duration{Duration: 300 * time.Second},
						AccessTokenMaxAgeSeconds:     216000,
					},
					Templates: configv1.OAuthTemplates{
						Login:             configv1.SecretNameReference{Name: "my-login-template"},
						ProviderSelection: configv1.SecretNameReference{Name: "provider-selection.template"},
						Error:             configv1.SecretNameReference{Name: "a.template-with-error"},
					},
				},
			},
		},
		{
			name: "two different IdPs in one object",
			args: args{
				spec: configv1.OAuthSpec{
					IdentityProviders: []configv1.IdentityProvider{
						{
							Name:                   "bad_bad_config",
							IdentityProviderConfig: doubledIdPs,
						},
					},
					TokenConfig: configv1.TokenConfig{
						AccessTokenMaxAgeSeconds: 216000,
					},
				},
			},
			want: field.ErrorList{
				field.Invalid(field.NewPath("spec", "identityProviders").Index(0), doubledIdPs, "only one identity provider can be configured in single object"),
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := validateOAuthSpec(tt.args.spec)

			// DeepEqual does not seem to be working well here
			var failedCheck bool
			if len(got) != len(tt.want) {
				failedCheck = true
			} else {
				// Check all the errors
				for i := range got {
					if got[i].Error() != tt.want[i].Error() {
						failedCheck = true
						break
					}
				}
			}

			if failedCheck {
				t.Errorf("validateOAuthSpec() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestValidateIdentityProvider(t *testing.T) {
	type args struct {
		identityProvider configv1.IdentityProvider
		fldPath          *field.Path
	}
	tests := []struct {
		name string
		args args
		want field.ErrorList
	}{
		{
			name: "empty provider needs at least name and type in provider",
			args: args{
				identityProvider: configv1.IdentityProvider{},
			},
			want: field.ErrorList{
				field.Required(field.NewPath("name"), ""),
				field.Required(field.NewPath("type"), ""),
			},
		},
		{
			name: "unknown type name",
			args: args{
				identityProvider: configv1.IdentityProvider{
					Name: "providingProvider",
					IdentityProviderConfig: configv1.IdentityProviderConfig{
						Type: "someText",
					},
				},
			},
			want: field.ErrorList{
				field.Invalid(field.NewPath("type"), "someText", "not a valid provider type"),
			},
		},
		{
			name: "basic provider",
			args: args{
				identityProvider: configv1.IdentityProvider{
					Name:                   "providingProvider",
					IdentityProviderConfig: htpasswdIDP(),
				},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ValidateIdentityProvider(tt.args.identityProvider, tt.args.fldPath)
			// DeepEqual does not seem to be working well here
			var failedCheck bool
			if len(got) != len(tt.want) {
				failedCheck = true
			} else {
				// Check all the errors
				for i := range got {
					if got[i].Error() != tt.want[i].Error() {
						failedCheck = true
						break
					}
				}
			}

			if failedCheck {
				t.Errorf("ValidateIdentityProvider() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestValidateOAuthIdentityProvider(t *testing.T) {
	type args struct {
		clientID        string
		clientSecretRef configv1.SecretNameReference
		fieldPath       *field.Path
	}
	tests := []struct {
		name string
		args args
		want field.ErrorList
	}{
		{
			name: "empty client ID and secret ref",
			args: args{
				clientID:        "",
				clientSecretRef: configv1.SecretNameReference{},
			},
			want: field.ErrorList{
				field.Required(field.NewPath("clientID"), ""),
				field.Required(field.NewPath("clientSecret", "name"), ""),
			},
		},
		{
			name: "improper client secret refname",
			args: args{
				clientID:        "thisBeClient",
				clientSecretRef: configv1.SecretNameReference{Name: "terribleName_forASecret"},
			},
			want: field.ErrorList{
				field.Invalid(field.NewPath("clientSecret", "name"), "terribleName_forASecret", wrongConfigMapSecretErrMsg),
			},
		},
		{
			name: "working example",
			args: args{
				clientID:        "thisBeClient",
				clientSecretRef: configv1.SecretNameReference{Name: "client-secret-hideout"},
			},
			want: field.ErrorList{},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := ValidateOAuthIdentityProvider(tt.args.clientID, tt.args.clientSecretRef, tt.args.fieldPath); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("ValidateOAuthIdentityProvider() = %v, want %v", got, tt.want)
			}
		})
	}
}
