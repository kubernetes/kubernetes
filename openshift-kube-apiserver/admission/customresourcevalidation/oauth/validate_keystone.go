package oauth

import (
	"k8s.io/apimachinery/pkg/util/validation/field"

	configv1 "github.com/openshift/api/config/v1"
)

func ValidateKeystoneIdentityProvider(provider *configv1.KeystoneIdentityProvider, fldPath *field.Path) field.ErrorList {
	errs := field.ErrorList{}
	if provider == nil {
		errs = append(errs, field.Required(fldPath, ""))
		return errs
	}

	errs = append(errs, ValidateRemoteConnectionInfo(provider.OAuthRemoteConnectionInfo, fldPath)...)

	if len(provider.DomainName) == 0 {
		errs = append(errs, field.Required(field.NewPath("domainName"), ""))
	}

	return errs
}
