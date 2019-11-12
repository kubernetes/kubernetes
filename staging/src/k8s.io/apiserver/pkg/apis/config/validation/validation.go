package validation

import (
	"fmt"

	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/apis/config"
)

var (
	allErrs   field.ErrorList
	root      = field.NewPath("EncryptionProviderConfiguration")
	resources = field.NewPath("EncryptionProviderConfiguration", "Resources")
)

func ValidateEncryptionConfig(config *config.EncryptionConfiguration) field.ErrorList {
	if config == nil {
		allErrs = append(
			allErrs,
			field.Required(
				root,
				"EncryptionProviderConfig can't be nil."))
		return allErrs
	}

	if len(config.Resources) == 0 {
		allErrs = append(
			allErrs,
			field.Required(resources,
				"EncryptionProviderConfiguration.Resources must contain at least one resource."))
		return allErrs
	}

	for i, r := range config.Resources {
		if r.Resources == nil || len(r.Resources) == 0 {
			allErrs = append(
				allErrs,
				field.Required(
					field.NewPath(
						"EncryptionProviderConfiguration",
						"Resources",
						"Resources"),
					fmt.Sprintf("EncryptionProviderConfiguration.Resources[%d] must contain at least one resource.", i)))

		}
		if r.Providers == nil || len(r.Providers) == 0 {
			allErrs = append(
				allErrs,
				field.Required(
					field.NewPath(
						"EncryptionProviderConfiguration",
						"Resources",
						"Providers"),
					fmt.Sprintf("EncryptionProviderConfiguration.Resources[%d] must contain at least one provider.", i)))
		}
	}

	return allErrs
}

func validateKMSConfig(config *config.KMSConfiguration) error {
	return nil
}

func validateAESConfig(config *config.AESConfiguration) error {
	return nil
}

func validateSecretBoxConfig(config *config.SecretboxConfiguration) error {
	return nil
}
