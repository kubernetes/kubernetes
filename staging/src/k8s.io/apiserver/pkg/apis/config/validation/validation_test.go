package validation

import (
	"fmt"
	"testing"

	"github.com/google/go-cmp/cmp"

	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/apis/config"
)

func TestValidateEncryptionConfigStructure(t *testing.T) {
	testCases := []struct {
		desc string
		in   *config.EncryptionConfiguration
		want field.ErrorList
	}{
		{
			desc: "nil config",
			in:   nil,
			want: field.ErrorList{encryptionProviderConfigShouldNotBeNil},
		},
		{
			desc: "empty config",
			in:   &config.EncryptionConfiguration{},
			want: field.ErrorList{atLeastOneResourceIsRequired},
		},
		{
			desc: "empty ResourceConfiguration",
			in: &config.EncryptionConfiguration{
				Resources: []config.ResourceConfiguration{
					{},
				},
			},
			want: field.ErrorList{
				field.Required(apiResources, fmt.Sprintf(atLeastOneK8SResourceRequiredFmt, 0)),
				field.Required(providers, fmt.Sprintf(atLeastOneProviderRequiredFmt, 0)),
			},
		},
		{
			desc: "empty provider config",
			in: &config.EncryptionConfiguration{
				Resources: []config.ResourceConfiguration{
					{
						Resources: []string{"secrets"},
						Providers: []config.ProviderConfiguration{},
					},
				},
			},
			want: field.ErrorList{field.Required(providers, fmt.Sprintf(atLeastOneProviderRequiredFmt, 0))},
		},
		{
			desc: "empty api resource",
			in: &config.EncryptionConfiguration{
				Resources: []config.ResourceConfiguration{
					{
						Resources: []string{},
						Providers: []config.ProviderConfiguration{
							{
								KMS: &config.KMSConfiguration{
									Name: "kms-provider",
								},
							},
						},
					},
				},
			},
			want: field.ErrorList{field.Required(apiResources, fmt.Sprintf(atLeastOneK8SResourceRequiredFmt, 0))},
		},
	}

	for _, tt := range testCases {
		t.Run(tt.desc, func(t *testing.T) {
			got := ValidateEncryptionConfiguration(tt.in)
			if d := cmp.Diff(tt.want, got); d != "" {
				t.Fatalf("ErrorList mismatch (-want +got):\n%s", d)
			}
		})
	}
}
