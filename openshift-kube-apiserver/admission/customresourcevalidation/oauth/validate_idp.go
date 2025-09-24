package oauth

import (
	"fmt"
	"strings"

	"k8s.io/apimachinery/pkg/api/validation/path"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation/field"
	pointerutil "k8s.io/utils/pointer"

	configv1 "github.com/openshift/api/config/v1"
	crvalidation "k8s.io/kubernetes/openshift-kube-apiserver/admission/customresourcevalidation"
)

const (
	// MinimumInactivityTimeoutSeconds defines the the smallest value allowed
	// for AccessTokenInactivityTimeoutSeconds.
	// It also defines the ticker interval for the token update routine as
	// MinimumInactivityTimeoutSeconds / 3 is used there.
	MinimumInactivityTimeoutSeconds = 5 * 60
)

var validMappingMethods = sets.NewString(
	string(configv1.MappingMethodLookup),
	string(configv1.MappingMethodClaim),
	string(configv1.MappingMethodAdd),
)

func validateOAuthSpec(spec configv1.OAuthSpec) field.ErrorList {
	errs := field.ErrorList{}
	specPath := field.NewPath("spec")

	providerNames := sets.NewString()

	challengeIssuingIdentityProviders := []string{}
	challengeRedirectingIdentityProviders := []string{}

	// TODO move to ValidateIdentityProviders (plural)
	for i, identityProvider := range spec.IdentityProviders {
		if isUsedAsChallenger(identityProvider.IdentityProviderConfig) {
			// TODO fix CAO to properly let you use request header and other challengers by disabling the other ones on CLI
			// RequestHeaderIdentityProvider is special, it can only react to challenge clients by redirecting them
			// Make sure we don't have more than a single redirector, and don't have a mix of challenge issuers and redirectors
			if identityProvider.Type == configv1.IdentityProviderTypeRequestHeader {
				challengeRedirectingIdentityProviders = append(challengeRedirectingIdentityProviders, identityProvider.Name)
			} else {
				challengeIssuingIdentityProviders = append(challengeIssuingIdentityProviders, identityProvider.Name)
			}
		}

		identityProviderPath := specPath.Child("identityProviders").Index(i)
		errs = append(errs, ValidateIdentityProvider(identityProvider, identityProviderPath)...)

		if len(identityProvider.Name) > 0 {
			if providerNames.Has(identityProvider.Name) {
				errs = append(errs, field.Invalid(identityProviderPath.Child("name"), identityProvider.Name, "must have a unique name"))
			}
			providerNames.Insert(identityProvider.Name)
		}
	}

	if len(challengeRedirectingIdentityProviders) > 1 {
		errs = append(errs, field.Invalid(specPath.Child("identityProviders"), "<omitted>", fmt.Sprintf("only one identity provider can redirect clients requesting an authentication challenge, found: %v", strings.Join(challengeRedirectingIdentityProviders, ", "))))
	}
	if len(challengeRedirectingIdentityProviders) > 0 && len(challengeIssuingIdentityProviders) > 0 {
		errs = append(errs, field.Invalid(specPath.Child("identityProviders"), "<omitted>", fmt.Sprintf(
			"cannot mix providers that redirect clients requesting auth challenges (%s) with providers issuing challenges to those clients (%s)",
			strings.Join(challengeRedirectingIdentityProviders, ", "),
			strings.Join(challengeIssuingIdentityProviders, ", "),
		)))
	}

	// TODO move to ValidateTokenConfig
	timeout := spec.TokenConfig.AccessTokenInactivityTimeout
	if timeout != nil && timeout.Seconds() < MinimumInactivityTimeoutSeconds {
		errs = append(errs, field.Invalid(
			specPath.Child("tokenConfig", "accessTokenInactivityTimeout"), timeout,
			fmt.Sprintf("the minimum acceptable token timeout value is %d seconds",
				MinimumInactivityTimeoutSeconds)))
	}

	if tokenMaxAge := spec.TokenConfig.AccessTokenMaxAgeSeconds; tokenMaxAge < 0 {
		errs = append(errs, field.Invalid(specPath.Child("tokenConfig", "accessTokenMaxAgeSeconds"), tokenMaxAge, "must be a positive integer or 0"))
	}

	// TODO move to ValidateTemplates
	errs = append(errs, crvalidation.ValidateSecretReference(specPath.Child("templates", "login"), spec.Templates.Login, false)...)
	errs = append(errs, crvalidation.ValidateSecretReference(specPath.Child("templates", "providerSelection"), spec.Templates.ProviderSelection, false)...)
	errs = append(errs, crvalidation.ValidateSecretReference(specPath.Child("templates", "error"), spec.Templates.Error, false)...)

	return errs
}

// if you change this, update the peer in user validation.  also, don't change this.
func validateIdentityProviderName(name string) []string {
	if reasons := path.ValidatePathSegmentName(name, false); len(reasons) != 0 {
		return reasons
	}

	if strings.Contains(name, ":") {
		return []string{`may not contain ":"`}
	}
	return nil
}

func ValidateIdentityProvider(identityProvider configv1.IdentityProvider, fldPath *field.Path) field.ErrorList {
	errs := field.ErrorList{}

	if len(identityProvider.Name) == 0 {
		errs = append(errs, field.Required(fldPath.Child("name"), ""))
	} else if reasons := validateIdentityProviderName(identityProvider.Name); len(reasons) != 0 {
		errs = append(errs, field.Invalid(fldPath.Child("name"), identityProvider.Name, strings.Join(reasons, ", ")))
	}

	if len(identityProvider.MappingMethod) > 0 && !validMappingMethods.Has(string(identityProvider.MappingMethod)) {
		errs = append(errs, field.NotSupported(fldPath.Child("mappingMethod"), identityProvider.MappingMethod, validMappingMethods.List()))
	}

	provider := identityProvider.IdentityProviderConfig
	// create a copy of the provider to simplify checking that only one IdPs is set
	providerCopy := provider.DeepCopy()
	switch provider.Type {
	case "":
		errs = append(errs, field.Required(fldPath.Child("type"), ""))

	case configv1.IdentityProviderTypeRequestHeader:
		errs = append(errs, ValidateRequestHeaderIdentityProvider(provider.RequestHeader, fldPath)...)
		providerCopy.RequestHeader = nil

	case configv1.IdentityProviderTypeBasicAuth:
		// TODO move to ValidateBasicAuthIdentityProvider for consistency
		if provider.BasicAuth == nil {
			errs = append(errs, field.Required(fldPath.Child("basicAuth"), ""))
		} else {
			errs = append(errs, ValidateRemoteConnectionInfo(provider.BasicAuth.OAuthRemoteConnectionInfo, fldPath.Child("basicAuth"))...)
		}
		providerCopy.BasicAuth = nil

	case configv1.IdentityProviderTypeHTPasswd:
		// TODO move to ValidateHTPasswdIdentityProvider for consistency
		if provider.HTPasswd == nil {
			errs = append(errs, field.Required(fldPath.Child("htpasswd"), ""))
		} else {
			errs = append(errs, crvalidation.ValidateSecretReference(fldPath.Child("htpasswd", "fileData"), provider.HTPasswd.FileData, true)...)
		}
		providerCopy.HTPasswd = nil

	case configv1.IdentityProviderTypeLDAP:
		errs = append(errs, ValidateLDAPIdentityProvider(provider.LDAP, fldPath.Child("ldap"))...)
		providerCopy.LDAP = nil

	case configv1.IdentityProviderTypeKeystone:
		errs = append(errs, ValidateKeystoneIdentityProvider(provider.Keystone, fldPath.Child("keystone"))...)
		providerCopy.Keystone = nil

	case configv1.IdentityProviderTypeGitHub:
		errs = append(errs, ValidateGitHubIdentityProvider(provider.GitHub, identityProvider.MappingMethod, fldPath.Child("github"))...)
		providerCopy.GitHub = nil

	case configv1.IdentityProviderTypeGitLab:
		errs = append(errs, ValidateGitLabIdentityProvider(provider.GitLab, fldPath.Child("gitlab"))...)
		providerCopy.GitLab = nil

	case configv1.IdentityProviderTypeGoogle:
		errs = append(errs, ValidateGoogleIdentityProvider(provider.Google, identityProvider.MappingMethod, fldPath.Child("google"))...)
		providerCopy.Google = nil

	case configv1.IdentityProviderTypeOpenID:
		errs = append(errs, ValidateOpenIDIdentityProvider(provider.OpenID, fldPath.Child("openID"))...)
		providerCopy.OpenID = nil

	default:
		errs = append(errs, field.Invalid(fldPath.Child("type"), identityProvider.Type, "not a valid provider type"))
	}

	if !pointerutil.AllPtrFieldsNil(providerCopy) {
		errs = append(errs, field.Invalid(fldPath, identityProvider.IdentityProviderConfig, "only one identity provider can be configured in single object"))
	}

	return errs
}

func ValidateOAuthIdentityProvider(clientID string, clientSecretRef configv1.SecretNameReference, fieldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	if len(clientID) == 0 {
		allErrs = append(allErrs, field.Required(fieldPath.Child("clientID"), ""))
	}

	allErrs = append(allErrs, crvalidation.ValidateSecretReference(fieldPath.Child("clientSecret"), clientSecretRef, true)...)

	return allErrs
}

func isUsedAsChallenger(idp configv1.IdentityProviderConfig) bool {
	// TODO this is wrong and needs to be more dynamic...
	switch idp.Type {
	// whitelist all the IdPs that we set `UseAsChallenger: true` in cluster-authentication-operator
	case configv1.IdentityProviderTypeBasicAuth, configv1.IdentityProviderTypeGitLab,
		configv1.IdentityProviderTypeHTPasswd, configv1.IdentityProviderTypeKeystone,
		configv1.IdentityProviderTypeLDAP,
		// guard open ID for now because it *could* have challenge in the future
		configv1.IdentityProviderTypeOpenID:
		return true
	case configv1.IdentityProviderTypeRequestHeader:
		if idp.RequestHeader == nil {
			// this is an error reported elsewhere
			return false
		}
		return len(idp.RequestHeader.ChallengeURL) > 0
	default:
		return false
	}
}
