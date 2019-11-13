package validation

import (
	"fmt"

	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/apis/config"
)

var (
	root = field.NewPath("EncryptionProviderConfiguration")
	// resources are encryption providers - not to be confused with k8s API resources.
	resources = field.NewPath(root.String(), "Resources")
	// apiResources are targets of encryption (ex. secrets).
	apiResources = field.NewPath(resources.String(), "Resources")
	providers    = field.NewPath(resources.String(), "Providers")

	encryptionProviderConfigShouldNotBeNil = field.Required(root, "EncryptionProviderConfig can't be nil.")
	atLeastOneResourceIsRequired           = field.Required(resources, "EncryptionProviderConfiguration.Resources must contain at least one resource.")

	atLeastOneK8SResourceRequiredFmt = "EncryptionProviderConfiguration.Resources[%d] must contain at least one resource."
	atLeastOneProviderRequiredFmt    = "EncryptionProviderConfiguration.Resources[%d] must contain at least one provider."
)

func ValidateEncryptionConfiguration(config *config.EncryptionConfiguration) field.ErrorList {
	var allErrs field.ErrorList

	if config == nil {
		return field.ErrorList{encryptionProviderConfigShouldNotBeNil}
	}

	if len(config.Resources) == 0 {
		return field.ErrorList{atLeastOneResourceIsRequired}
	}

	for i, r := range config.Resources {
		allErrs = append(allErrs, validateEncryptionConfigurationResource(r, i)...)
	}

	return allErrs
}

func validateEncryptionConfigurationResource(resource config.ResourceConfiguration, idx int) field.ErrorList {
	var allErrs field.ErrorList
	if resource.Resources == nil || len(resource.Resources) == 0 {
		allErrs = append(allErrs, field.Required(apiResources, fmt.Sprintf(atLeastOneK8SResourceRequiredFmt, idx)))
	}
	if resource.Providers == nil || len(resource.Providers) == 0 {
		allErrs = append(allErrs, field.Required(providers, fmt.Sprintf(atLeastOneProviderRequiredFmt, idx)))
	}

	for i, p := range resource.Providers {
		found := false
		if p.KMS != nil {
			allErrs = append(allErrs, validateKMSConfig(p.KMS, i)...)
			found = true
		}
		if p.AESCBC != nil {
			if found {
				allErrs = append(allErrs, multipleProvidersError(p, i))
			}
			allErrs = append(allErrs, validateAESCBCConfig(p.AESCBC, i)...)
			found = true
		}
		if p.AESGCM != nil {
			if found {
				allErrs = append(allErrs, multipleProvidersError(p, i))
			}
			allErrs = append(allErrs, validateAESGCMConfig(p.AESGCM, i)...)
			found = true
		}
		if p.Secretbox != nil {
			if found {
				allErrs = append(allErrs, multipleProvidersError(p, i))
			}
			allErrs = append(allErrs, validateSecretBoxConfig(p.Secretbox, i)...)
			found = true
		}
		if p.Identity != nil {
			if found {
				allErrs = append(allErrs, multipleProvidersError(p, i))
			}
			allErrs = append(allErrs, validateIdentityConfig(p.Identity, i)...)
			found = true
		}

		if !found {
			allErrs = append(
				allErrs,
				field.NotSupported(field.NewPath(providers.String(),
					fmt.Sprintf("[%d]", i)),
					p,
					[]string{"KMS", "AESCBC", "AESGCM", "Secretbox", "Identity"}))
		}

	}

	return allErrs
}

func multipleProvidersError(provider config.ProviderConfiguration, idx int) *field.Error {
	return field.NotSupported(
		field.NewPath(providers.String(), fmt.Sprintf("[%d]", idx)),
		provider,
		[]string{"KMS", "AESCBC", "AESGCM", "Secretbox", "Identity"})
}

func validateKMSConfig(config *config.KMSConfiguration, resourceIdx int) field.ErrorList {
	return nil
}

func validateAESCBCConfig(config *config.AESConfiguration, resourceIdx int) field.ErrorList {
	return nil
}

func validateAESGCMConfig(config *config.AESConfiguration, resourceIdx int) field.ErrorList {
	return nil
}

func validateSecretBoxConfig(config *config.SecretboxConfiguration, resourceIdx int) field.ErrorList {
	return nil
}

func validateIdentityConfig(config *config.IdentityConfiguration, resourceIdx int) field.ErrorList {
	return nil
}
