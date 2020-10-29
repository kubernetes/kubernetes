package oauth

import (
	"reflect"
	"testing"

	configv1 "github.com/openshift/api/config/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

func googleIDP() configv1.IdentityProviderConfig {
	return configv1.IdentityProviderConfig{
		Type: configv1.IdentityProviderTypeGoogle,
		Google: &configv1.GoogleIdentityProvider{
			ClientID:     "masterOfInstances",
			ClientSecret: configv1.SecretNameReference{Name: "secret-google-secret"},
			HostedDomain: "myprivategoogledomain.com",
		},
	}
}

func TestValidateGoogleIdentityProvider(t *testing.T) {
	type args struct {
		provider      *configv1.GoogleIdentityProvider
		mappingMethod configv1.MappingMethodType
		fieldPath     *field.Path
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
				provider: &configv1.GoogleIdentityProvider{
					HostedDomain: "myprivategoogledomain.com",
				},
			},
			want: field.ErrorList{
				field.Required(field.NewPath("clientID"), ""),
				field.Required(field.NewPath("clientSecret", "name"), ""),
			},
		},
		{
			name: "no hosted domain with mapping method != 'lookup'",
			args: args{
				provider: &configv1.GoogleIdentityProvider{
					ClientID:     "masterOfInstances",
					ClientSecret: configv1.SecretNameReference{Name: "secret-google-secret"},
				},
				mappingMethod: configv1.MappingMethodClaim,
			},
			want: field.ErrorList{
				field.Invalid(field.NewPath("hostedDomain"), nil, "hostedDomain must be specified unless lookup is used"),
			},
		},
		{
			name: "no hosted domain with mapping method == 'lookup'",
			args: args{
				provider: &configv1.GoogleIdentityProvider{
					ClientID:     "masterOfInstances",
					ClientSecret: configv1.SecretNameReference{Name: "secret-google-secret"},
				},
				mappingMethod: configv1.MappingMethodLookup,
			},
			want: field.ErrorList{},
		},
		{
			name: "working example",
			args: args{
				provider: googleIDP().Google,
			},
			want: field.ErrorList{},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := ValidateGoogleIdentityProvider(tt.args.provider, tt.args.mappingMethod, tt.args.fieldPath); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("ValidateGoogleIdentityProvider() = %v, want %v", got, tt.want)
			}
		})
	}
}
