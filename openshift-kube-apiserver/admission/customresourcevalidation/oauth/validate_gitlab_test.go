package oauth

import (
	"reflect"
	"testing"

	configv1 "github.com/openshift/api/config/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

func gitlabIDP() configv1.IdentityProviderConfig {
	return configv1.IdentityProviderConfig{
		Type: configv1.IdentityProviderTypeGitLab,
		GitLab: &configv1.GitLabIdentityProvider{
			ClientID:     "masterOfInstances",
			ClientSecret: configv1.SecretNameReference{Name: "secret-gitlab-secret"},
			URL:          "https://thisgitlabinstancerighthere.com",
			CA:           configv1.ConfigMapNameReference{Name: "letsencrypt-for-gitlab.instance"},
		},
	}
}

func TestValidateGitLabIdentityProvider(t *testing.T) {
	type args struct {
		provider  *configv1.GitLabIdentityProvider
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
			name: "insecure URL",
			args: args{
				provider: &configv1.GitLabIdentityProvider{
					ClientID:     "hereBeMyId",
					ClientSecret: configv1.SecretNameReference{Name: "gitlab-client-sec"},
					URL:          "http://anyonecanseemenow.com",
				},
			},
			want: field.ErrorList{
				field.Invalid(field.NewPath("url"), "http://anyonecanseemenow.com", "must use https scheme"),
			},
		},
		{
			name: "missing client ID and secret",
			args: args{
				provider: &configv1.GitLabIdentityProvider{
					URL: "https://privategitlab.com",
				},
			},
			want: field.ErrorList{
				field.Required(field.NewPath("clientID"), ""),
				field.Required(field.NewPath("clientSecret", "name"), ""),
			},
		},
		{
			name: "invalid CA ref name",
			args: args{
				provider: &configv1.GitLabIdentityProvider{
					ClientID:     "hereBeMyId",
					ClientSecret: configv1.SecretNameReference{Name: "gitlab-client-sec"},
					URL:          "https://anyonecanseemenow.com",
					CA:           configv1.ConfigMapNameReference{Name: "veryBadRefName?:("},
				},
			},
			want: field.ErrorList{
				field.Invalid(field.NewPath("ca", "name"), "veryBadRefName?:(", wrongConfigMapSecretErrMsg),
			},
		},
		{
			name: "minimal passing case",
			args: args{
				provider: &configv1.GitLabIdentityProvider{
					ClientID:     "hereBeMyId",
					ClientSecret: configv1.SecretNameReference{Name: "gitlab-client-sec"},
					URL:          "https://anyonecanseemenow.com",
				},
			},
			want: field.ErrorList{},
		},
		{
			name: "more complicated case",
			args: args{
				provider: gitlabIDP().GitLab,
			},
			want: field.ErrorList{},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := ValidateGitLabIdentityProvider(tt.args.provider, tt.args.fieldPath); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("ValidateGitLabIdentityProvider() = %v, want %v", got, tt.want)
			}
		})
	}
}
