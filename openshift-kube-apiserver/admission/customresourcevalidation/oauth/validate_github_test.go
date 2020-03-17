package oauth

import (
	"reflect"
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"

	configv1 "github.com/openshift/api/config/v1"
)

func TestValidateGitHubIdentityProvider(t *testing.T) {
	type args struct {
		provider      *configv1.GitHubIdentityProvider
		mappingMethod configv1.MappingMethodType
		fieldPath     *field.Path
	}
	tests := []struct {
		name   string
		args   args
		errors field.ErrorList
	}{
		{
			name: "cannot use GH as hostname",
			args: args{
				provider: &configv1.GitHubIdentityProvider{
					ClientID:      "client",
					ClientSecret:  configv1.SecretNameReference{Name: "secret"},
					Organizations: []string{"org1"},
					Teams:         nil,
					Hostname:      "github.com",
					CA:            configv1.ConfigMapNameReference{Name: "caconfigmap"},
				},
				mappingMethod: "",
			},
			errors: field.ErrorList{
				{Type: field.ErrorTypeInvalid, Field: "hostname", BadValue: "github.com", Detail: "cannot equal [*.]github.com"},
			},
		},
		{
			name: "cannot use GH subdomain as hostname",
			args: args{
				provider: &configv1.GitHubIdentityProvider{
					ClientID:      "client",
					ClientSecret:  configv1.SecretNameReference{Name: "secret"},
					Organizations: []string{"org1"},
					Teams:         nil,
					Hostname:      "foo.github.com",
					CA:            configv1.ConfigMapNameReference{Name: "caconfigmap"},
				},
				mappingMethod: "",
			},
			errors: field.ErrorList{
				{Type: field.ErrorTypeInvalid, Field: "hostname", BadValue: "foo.github.com", Detail: "cannot equal [*.]github.com"},
			},
		},
		{
			name: "valid domain hostname",
			args: args{
				provider: &configv1.GitHubIdentityProvider{
					ClientID:      "client",
					ClientSecret:  configv1.SecretNameReference{Name: "secret"},
					Organizations: []string{"org1"},
					Teams:         nil,
					Hostname:      "company.com",
					CA:            configv1.ConfigMapNameReference{Name: "caconfigmap"},
				},
				mappingMethod: "",
			},
		},
		{
			name: "valid ip hostname",
			args: args{
				provider: &configv1.GitHubIdentityProvider{
					ClientID:      "client",
					ClientSecret:  configv1.SecretNameReference{Name: "secret"},
					Organizations: []string{"org1"},
					Teams:         nil,
					Hostname:      "192.168.8.1",
					CA:            configv1.ConfigMapNameReference{Name: "caconfigmap"},
				},
				mappingMethod: "",
			},
		},
		{
			name: "invalid ip hostname with port",
			args: args{
				provider: &configv1.GitHubIdentityProvider{
					ClientID:      "client",
					ClientSecret:  configv1.SecretNameReference{Name: "secret"},
					Organizations: []string{"org1"},
					Teams:         nil,
					Hostname:      "192.168.8.1:8080",
					CA:            configv1.ConfigMapNameReference{Name: "caconfigmap"},
				},
				mappingMethod: "",
			},
			errors: field.ErrorList{
				{Type: field.ErrorTypeInvalid, Field: "hostname", BadValue: "192.168.8.1:8080", Detail: "must be a valid DNS subdomain or IP address"},
			},
		},
		{
			name: "invalid domain hostname",
			args: args{
				provider: &configv1.GitHubIdentityProvider{
					ClientID:      "client",
					ClientSecret:  configv1.SecretNameReference{Name: "secret"},
					Organizations: []string{"org1"},
					Teams:         nil,
					Hostname:      "google-.com",
					CA:            configv1.ConfigMapNameReference{Name: "caconfigmap"},
				},
				mappingMethod: "",
			},
			errors: field.ErrorList{
				{Type: field.ErrorTypeInvalid, Field: "hostname", BadValue: "google-.com", Detail: "must be a valid DNS subdomain or IP address"},
			},
		},
		{
			name: "invalid name in ca ref and no hostname",
			args: args{
				provider: &configv1.GitHubIdentityProvider{
					ClientID:      "client",
					ClientSecret:  configv1.SecretNameReference{Name: "secret"},
					Organizations: []string{"org1"},
					Teams:         nil,
					Hostname:      "",
					CA:            configv1.ConfigMapNameReference{Name: "ca&config-map"},
				},
				mappingMethod: "",
			},
			errors: field.ErrorList{
				{Type: field.ErrorTypeInvalid, Field: "ca.name", BadValue: "ca&config-map", Detail: "a DNS-1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')"},
				{Type: field.ErrorTypeInvalid, Field: "ca", BadValue: configv1.ConfigMapNameReference{Name: "ca&config-map"}, Detail: "cannot be specified when hostname is empty"},
			},
		},
		{
			name: "valid ca and hostname",
			args: args{
				provider: &configv1.GitHubIdentityProvider{
					ClientID:      "client",
					ClientSecret:  configv1.SecretNameReference{Name: "secret"},
					Organizations: []string{"org1"},
					Teams:         nil,
					Hostname:      "mo.co",
					CA:            configv1.ConfigMapNameReference{Name: "ca-config-map"},
				},
				mappingMethod: "",
			},
		},
		{
			name: "GitHub requires client ID and secret",
			args: args{
				provider: &configv1.GitHubIdentityProvider{
					ClientID:      "",
					ClientSecret:  configv1.SecretNameReference{},
					Organizations: []string{"org1"},
					Teams:         nil,
					Hostname:      "",
					CA:            configv1.ConfigMapNameReference{},
				},
				mappingMethod: "",
			},
			errors: field.ErrorList{
				{Type: field.ErrorTypeRequired, Field: "provider.clientID", BadValue: "", Detail: ""},
				{Type: field.ErrorTypeRequired, Field: "provider.clientSecret.name", BadValue: "", Detail: ""},
			},
		},
		{
			name: "GitHub warns when not constrained to organizations or teams without lookup",
			args: args{
				provider: &configv1.GitHubIdentityProvider{
					ClientID:      "client",
					ClientSecret:  configv1.SecretNameReference{Name: "secret"},
					Organizations: nil,
					Teams:         nil,
					Hostname:      "",
					CA:            configv1.ConfigMapNameReference{},
				},
				mappingMethod: "",
			},
			errors: field.ErrorList{
				{Type: field.ErrorTypeInvalid, Field: "", BadValue: nil, Detail: "one of organizations or teams must be specified unless hostname is set or lookup is used"},
			},
		},
		{
			name: "GitHub does not warn when not constrained to organizations or teams with lookup",
			args: args{
				provider: &configv1.GitHubIdentityProvider{
					ClientID:      "client",
					ClientSecret:  configv1.SecretNameReference{Name: "secret"},
					Organizations: nil,
					Teams:         nil,
					Hostname:      "",
					CA:            configv1.ConfigMapNameReference{},
				},
				mappingMethod: "lookup",
			},
		},
		{
			name: "invalid cannot specific both organizations and teams",
			args: args{
				provider: &configv1.GitHubIdentityProvider{
					ClientID:      "client",
					ClientSecret:  configv1.SecretNameReference{Name: "secret"},
					Organizations: []string{"org1"},
					Teams:         []string{"org1/team1"},
					Hostname:      "",
					CA:            configv1.ConfigMapNameReference{},
				},
				mappingMethod: "",
			},
			errors: field.ErrorList{
				{Type: field.ErrorTypeInvalid, Field: "organizations", BadValue: []string{"org1"}, Detail: "specify organizations or teams, not both"},
				{Type: field.ErrorTypeInvalid, Field: "teams", BadValue: []string{"org1/team1"}, Detail: "specify organizations or teams, not both"},
			},
		},
		{
			name: "invalid team format",
			args: args{
				provider: &configv1.GitHubIdentityProvider{
					ClientID:      "client",
					ClientSecret:  configv1.SecretNameReference{Name: "secret"},
					Organizations: nil,
					Teams:         []string{"org1/team1", "org2/not/team2", "org3//team3", "", "org4/team4"},
					Hostname:      "",
					CA:            configv1.ConfigMapNameReference{},
				},
				mappingMethod: "",
			},
			errors: field.ErrorList{
				{Type: field.ErrorTypeInvalid, Field: "teams[1]", BadValue: "org2/not/team2", Detail: "must be in the format <org>/<team>"},
				{Type: field.ErrorTypeInvalid, Field: "teams[2]", BadValue: "org3//team3", Detail: "must be in the format <org>/<team>"},
				{Type: field.ErrorTypeInvalid, Field: "teams[3]", BadValue: "", Detail: "must be in the format <org>/<team>"},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ValidateGitHubIdentityProvider(tt.args.provider, tt.args.mappingMethod, tt.args.fieldPath)
			if tt.errors == nil && len(got) == 0 {
				return
			}
			if !reflect.DeepEqual(got, tt.errors) {
				t.Errorf("ValidateGitHubIdentityProvider() = %v, want %v", got, tt.errors)
			}
		})
	}
}
