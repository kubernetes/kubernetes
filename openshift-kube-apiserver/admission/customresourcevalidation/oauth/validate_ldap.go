package oauth

import (
	"fmt"

	"k8s.io/apimachinery/pkg/util/validation/field"

	configv1 "github.com/openshift/api/config/v1"
	"github.com/openshift/library-go/pkg/security/ldaputil"
	crvalidation "k8s.io/kubernetes/openshift-kube-apiserver/admission/customresourcevalidation"
)

func ValidateLDAPIdentityProvider(provider *configv1.LDAPIdentityProvider, fldPath *field.Path) field.ErrorList {
	errs := field.ErrorList{}

	if provider == nil {
		errs = append(errs, field.Required(fldPath, ""))
		return errs
	}

	errs = append(errs, validateLDAPClientConfig(provider.URL, provider.BindDN, provider.BindPassword.Name, provider.CA.Name, provider.Insecure, fldPath)...)
	errs = append(errs, crvalidation.ValidateSecretReference(fldPath.Child("bindPassword"), provider.BindPassword, false)...)
	errs = append(errs, crvalidation.ValidateConfigMapReference(fldPath.Child("ca"), provider.CA, false)...)

	// At least one attribute to use as the user id is required
	if len(provider.Attributes.ID) == 0 {
		errs = append(errs, field.Invalid(fldPath.Child("attributes", "id"), "[]", "at least one id attribute is required (LDAP standard identity attribute is 'dn')"))
	}

	return errs
}

// TODO clean this up
func validateLDAPClientConfig(url, bindDN, bindPasswordRef, CA string, insecure bool, fldPath *field.Path) field.ErrorList {
	errs := field.ErrorList{}

	// Make sure bindDN and bindPassword are both set, or both unset
	// Both unset means an anonymous bind is used for search (https://tools.ietf.org/html/rfc4513#section-5.1.1)
	// Both set means the name/password simple bind is used for search (https://tools.ietf.org/html/rfc4513#section-5.1.3)
	if (len(bindDN) == 0) != (len(bindPasswordRef) == 0) {
		errs = append(errs, field.Invalid(fldPath.Child("bindDN"), bindDN, "bindDN and bindPassword must both be specified, or both be empty"))
		errs = append(errs, field.Invalid(fldPath.Child("bindPassword").Child("name"), bindPasswordRef, "bindDN and bindPassword must both be specified, or both be empty"))
	}

	if len(url) == 0 {
		errs = append(errs, field.Required(fldPath.Child("url"), ""))
		return errs
	}

	u, err := ldaputil.ParseURL(url)
	if err != nil {
		errs = append(errs, field.Invalid(fldPath.Child("url"), url, err.Error()))
		return errs
	}

	if insecure {
		if u.Scheme == ldaputil.SchemeLDAPS {
			errs = append(errs, field.Invalid(fldPath.Child("url"), url, fmt.Sprintf("Cannot use %s scheme with insecure=true", u.Scheme)))
		}
		if len(CA) > 0 {
			errs = append(errs, field.Invalid(fldPath.Child("ca"), CA, "Cannot specify a ca with insecure=true"))
		}
	}

	return errs
}
