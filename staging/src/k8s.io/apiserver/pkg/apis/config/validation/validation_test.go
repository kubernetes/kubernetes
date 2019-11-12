package validation

import (
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
			want: field.ErrorList{
				field.Required(field.NewPath("EncryptionProviderConfiguration"), "EncryptionProviderConfig can't be nil."),
			},
		},
		{
			desc: "empty config",
			in:   &config.EncryptionConfiguration{},
			want: field.ErrorList{
				field.Required(field.NewPath("EncryptionProviderConfiguration", "Resources"), "EncryptionProviderConfiguration.Resources must contain at least one resource."),
			},
		},
		{
			desc: "empty ResourceConfiguration",
			in: &config.EncryptionConfiguration{
				Resources: []config.ResourceConfiguration{
					{},
				},
			},
			want: field.ErrorList{
				field.Required(
					field.NewPath(
						"EncryptionProviderConfiguration",
						"Resources",
						"Resources"),
					"EncryptionProviderConfiguration.Resources[0] must contain at least one resource."),
				field.Required(
					field.NewPath(
						"EncryptionProviderConfiguration",
						"Resources",
						"Providers"),
					"EncryptionProviderConfiguration.Resources[0] must contain at least one provider."),
			},
		},
	}

	for _, tt := range testCases {
		t.Run(tt.desc, func(t *testing.T) {
			got := ValidateEncryptionConfig(tt.in)
			if d := cmp.Diff(tt.want, got); d != "" {
				t.Fatalf("ErrorList mismatch (-want +got):\n%s", d)
			}
		})
	}
}
