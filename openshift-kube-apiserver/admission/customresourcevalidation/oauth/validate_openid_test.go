package oauth

import (
	"reflect"
	"testing"

	configv1 "github.com/openshift/api/config/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

func openidIDP() configv1.IdentityProviderConfig {
	return configv1.IdentityProviderConfig{
		Type: configv1.IdentityProviderTypeOpenID,
		OpenID: &configv1.OpenIDIdentityProvider{
			ClientID:     "readallPerson",
			ClientSecret: configv1.SecretNameReference{Name: "oidc-secret"},
			Issuer:       "https://oidc-friendly.domain.com",
			CA:           configv1.ConfigMapNameReference{Name: "oidc-ca"},
			ExtraScopes:  []string{"email", "profile"},
			ExtraAuthorizeParameters: map[string]string{
				"include_granted_scopes": "true",
			},
			Claims: configv1.OpenIDClaims{
				PreferredUsername: []string{"full_name", "email"},
				Email:             []string{"email"},
			},
		},
	}
}

func TestValidateOpenIDIdentityProvider(t *testing.T) {
	type args struct {
		provider  *configv1.OpenIDIdentityProvider
		fieldPath *field.Path
	}
	tests := []struct {
		name string
		args args
		want field.ErrorList
	}{
		{
			name: "nil input provider",
			want: field.ErrorList{
				field.Required(nil, ""),
			},
		},
		{
			name: "missing client ID and secret",
			args: args{
				provider: &configv1.OpenIDIdentityProvider{
					Issuer: "https://bigcorp.oidc.com",
				},
			},
			want: field.ErrorList{
				field.Required(field.NewPath("clientID"), ""),
				field.Required(field.NewPath("clientSecret", "name"), ""),
			},
		},
		{
			name: "missing issuer",
			args: args{
				provider: &configv1.OpenIDIdentityProvider{
					ClientID:     "readallPerson",
					ClientSecret: configv1.SecretNameReference{Name: "oidc-secret"},
				},
			},
			want: field.ErrorList{
				field.Invalid(field.NewPath("issuer"), "", "must contain a scheme (e.g. https://)"),
				field.Invalid(field.NewPath("issuer"), "", "must contain a host"),
			},
		},
		{
			name: "issuer with http:// scheme",
			args: args{
				provider: &configv1.OpenIDIdentityProvider{
					ClientID:     "gentleDolphin",
					ClientSecret: configv1.SecretNameReference{Name: "seemsliggit"},
					Issuer:       "http://oidc-friendly.domain.com",
				},
			},
			want: field.ErrorList{
				field.Invalid(field.NewPath("issuer"), "http://oidc-friendly.domain.com", "must use https scheme"),
			},
		},
		{
			name: "bad CA refname",
			args: args{
				provider: &configv1.OpenIDIdentityProvider{
					ClientID:     "readallPerson",
					ClientSecret: configv1.SecretNameReference{Name: "oidc-secret"},
					Issuer:       "https://oidc-friendly.domain.com",
					CA:           configv1.ConfigMapNameReference{Name: "the_Nameofaca"},
				},
			},
			want: field.ErrorList{
				field.Invalid(field.NewPath("ca", "name"), "the_Nameofaca", wrongConfigMapSecretErrMsg),
			},
		},
		{
			name: "minimal working example",
			args: args{
				provider: &configv1.OpenIDIdentityProvider{
					ClientID:     "readallPerson",
					ClientSecret: configv1.SecretNameReference{Name: "oidc-secret"},
					Issuer:       "https://oidc-friendly.domain.com",
				},
			},
			want: field.ErrorList{},
		},
		{
			name: "more complicated use",
			args: args{
				provider: openidIDP().OpenID,
			},
			want: field.ErrorList{},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := ValidateOpenIDIdentityProvider(tt.args.provider, tt.args.fieldPath); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("ValidateOpenIDIdentityProvider() = %v, want %v", got, tt.want)
			}
		})
	}
}
