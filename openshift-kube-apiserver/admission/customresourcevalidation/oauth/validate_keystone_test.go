package oauth

import (
	"reflect"
	"testing"

	configv1 "github.com/openshift/api/config/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

func keystoneIdP() configv1.IdentityProviderConfig {
	return configv1.IdentityProviderConfig{
		Type: configv1.IdentityProviderTypeKeystone,
		Keystone: &configv1.KeystoneIdentityProvider{
			OAuthRemoteConnectionInfo: configv1.OAuthRemoteConnectionInfo{
				URL: "https://somewhere.over.rainbow/ks",
				CA:  configv1.ConfigMapNameReference{Name: "govt-ca"},
			},
			DomainName: "production",
		},
	}
}

func TestValidateKeystoneIdentityProvider(t *testing.T) {
	type args struct {
		provider *configv1.KeystoneIdentityProvider
		fldPath  *field.Path
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
			name: "empty url",
			args: args{
				provider: &configv1.KeystoneIdentityProvider{
					OAuthRemoteConnectionInfo: configv1.OAuthRemoteConnectionInfo{
						URL: "",
					},
					DomainName: "production",
				},
			},
			want: field.ErrorList{
				field.Required(field.NewPath("url"), ""),
			},
		},
		{
			name: "http url",
			args: args{
				provider: &configv1.KeystoneIdentityProvider{
					OAuthRemoteConnectionInfo: configv1.OAuthRemoteConnectionInfo{
						URL: "http://foo",
					},
					DomainName: "production",
				},
			},
			want: field.ErrorList{
				field.Invalid(field.NewPath("url"), "http://foo", "must use https scheme"),
			},
		},
		{
			name: "missing domain name",
			args: args{
				provider: &configv1.KeystoneIdentityProvider{
					OAuthRemoteConnectionInfo: configv1.OAuthRemoteConnectionInfo{
						URL: "https://keystone.openstack.nasa.gov/",
					},
				},
			},
			want: field.ErrorList{
				field.Required(field.NewPath("domainName"), ""),
			},
		},
		{
			name: "working provider",
			args: args{
				provider: keystoneIdP().Keystone,
			},
			want: field.ErrorList{},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := ValidateKeystoneIdentityProvider(tt.args.provider, tt.args.fldPath); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("ValidateKeystoneIdentityProvider() = %v, want %v", got, tt.want)
			}
		})
	}
}
