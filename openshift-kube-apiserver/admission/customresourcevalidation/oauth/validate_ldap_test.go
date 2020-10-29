package oauth

import (
	"reflect"
	"testing"

	configv1 "github.com/openshift/api/config/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

func ldapIDP() configv1.IdentityProviderConfig {
	return configv1.IdentityProviderConfig{
		Type: configv1.IdentityProviderTypeLDAP,
		LDAP: &configv1.LDAPIdentityProvider{
			Attributes: configv1.LDAPAttributeMapping{
				ID: []string{"memberUid"},
			},
			BindDN: "uid=readallaccount,ou=privileged,dc=coolpeople,dc=se",
			BindPassword: configv1.SecretNameReference{
				Name: "ldap-secret",
			},
			CA:       configv1.ConfigMapNameReference{Name: "ldap-ca-configmap"},
			Insecure: false,
			URL:      "ldaps://ldapinstance.corporate.coolpeople.se/ou=Groups,dc=coolpeople,dc=se?memberUid?sub",
		},
	}
}

func TestValidateLDAPIdentityProvider(t *testing.T) {
	type args struct {
		provider *configv1.LDAPIdentityProvider
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
			name: "invalid bindPassword ref name, missing ID",
			args: args{
				provider: &configv1.LDAPIdentityProvider{
					BindPassword: configv1.SecretNameReference{Name: "bad_refname"},
				},
			},
			want: field.ErrorList{
				field.Invalid(field.NewPath("bindDN"), "", "bindDN and bindPassword must both be specified, or both be empty"),
				field.Invalid(field.NewPath("bindPassword", "name"), "bad_refname", "bindDN and bindPassword must both be specified, or both be empty"),
				field.Required(field.NewPath("url"), ""),
				field.Invalid(field.NewPath("bindPassword", "name"), "bad_refname", wrongConfigMapSecretErrMsg),
				field.Invalid(field.NewPath("attributes", "id"), "[]", "at least one id attribute is required (LDAP standard identity attribute is 'dn')"),
			},
		},
		{
			name: "invalid url",
			args: args{
				provider: &configv1.LDAPIdentityProvider{
					URL: "https://foo",
					Attributes: configv1.LDAPAttributeMapping{
						ID: []string{"uid"},
					},
				},
			},
			want: field.ErrorList{
				field.Invalid(field.NewPath("url"), "https://foo", `invalid scheme "https"`),
			},
		},
		{
			name: "minimal passing provider",
			args: args{
				provider: &configv1.LDAPIdentityProvider{
					URL: "ldap://foo",
					Attributes: configv1.LDAPAttributeMapping{
						ID: []string{"uid"},
					},
				},
			},
			want: field.ErrorList{},
		},
		{
			name: "more complicated use",
			args: args{
				provider: ldapIDP().LDAP,
			},
			want: field.ErrorList{},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := ValidateLDAPIdentityProvider(tt.args.provider, tt.args.fldPath); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("ValidateLDAPIdentityProvider() = %v, want %v", got, tt.want)
			}
		})
	}
}
