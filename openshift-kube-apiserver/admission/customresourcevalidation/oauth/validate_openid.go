package oauth

import (
	"fmt"
	"strings"

	"k8s.io/apimachinery/pkg/util/validation/field"

	configv1 "github.com/openshift/api/config/v1"
	"github.com/openshift/library-go/pkg/config/validation"
	crvalidation "k8s.io/kubernetes/openshift-kube-apiserver/admission/customresourcevalidation"
)

func ValidateOpenIDIdentityProvider(provider *configv1.OpenIDIdentityProvider, fieldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if provider == nil {
		allErrs = append(allErrs, field.Required(fieldPath, ""))
		return allErrs
	}

	allErrs = append(allErrs, ValidateOAuthIdentityProvider(provider.ClientID, provider.ClientSecret, fieldPath)...)

	if provider.Issuer != strings.TrimRight(provider.Issuer, "/") {
		allErrs = append(allErrs, field.Invalid(fieldPath.Child("issuer"), provider.Issuer, "cannot end with '/'"))
	}

	// The specs are a bit ambiguous on whether this must or needn't be https://
	// schema, but they do require (MUST) TLS support for the discovery and we do
	// require this in out API description
	// https://openid.net/specs/openid-connect-discovery-1_0.html#TLSRequirements
	url, issuerErrs := validation.ValidateSecureURL(provider.Issuer, fieldPath.Child("issuer"))
	allErrs = append(allErrs, issuerErrs...)
	if len(url.RawQuery) > 0 || len(url.Fragment) > 0 {
		allErrs = append(allErrs, field.Invalid(fieldPath.Child("issuer"), provider.Issuer, "must not specify query or fragment component"))
	}

	allErrs = append(allErrs, crvalidation.ValidateConfigMapReference(fieldPath.Child("ca"), provider.CA, false)...)

	for i, scope := range provider.ExtraScopes {
		// https://tools.ietf.org/html/rfc6749#section-3.3 (full list of allowed chars is %x21 / %x23-5B / %x5D-7E)
		// for those without an ascii table, that's `!`, `#-[`, `]-~` inclusive.
		for _, ch := range scope {
			switch {
			case ch == '!':
			case ch >= '#' && ch <= '[':
			case ch >= ']' && ch <= '~':
			default:
				allErrs = append(allErrs, field.Invalid(fieldPath.Child("extraScopes").Index(i), scope, fmt.Sprintf("cannot contain %v", ch)))
			}
		}
	}

	return allErrs
}
