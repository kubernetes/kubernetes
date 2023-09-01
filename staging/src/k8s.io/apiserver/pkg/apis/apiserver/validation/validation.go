/*
Copyright 2023 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package validation

import (
	"fmt"
	"net/url"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation/field"
	api "k8s.io/apiserver/pkg/apis/apiserver"
	"k8s.io/client-go/util/cert"
)

const (
	atLeastOneRequiredErrFmt = "at least one %s is required"
)

var (
	root = field.NewPath("jwt")
)

// ValidateAuthenticationConfiguration validates a given AuthenticationConfiguration.
func ValidateAuthenticationConfiguration(c *api.AuthenticationConfiguration) field.ErrorList {
	var allErrs field.ErrorList

	// This stricter validation is solely based on what the current implementation supports.
	// TODO(aramase): when StructuredAuthenticationConfiguration feature gate is added and wired up,
	// relax this check to allow 0 authenticators. This will allow us to support the case where
	// API server is initially configured with no authenticators and then authenticators are added
	// later via dynamic config.
	if len(c.JWT) == 0 {
		allErrs = append(allErrs, field.Required(root, fmt.Sprintf(atLeastOneRequiredErrFmt, root)))
		return allErrs
	}

	// This stricter validation is because the --oidc-* flag option is singular.
	// TODO(aramase): when StructuredAuthenticationConfiguration feature gate is added and wired up,
	// remove the 1 authenticator limit check and add set the limit to 64.
	if len(c.JWT) > 1 {
		allErrs = append(allErrs, field.TooMany(root, len(c.JWT), 1))
		return allErrs
	}

	// TODO(aramase): right now we only support a single JWT authenticator as
	// this is wired to the --oidc-* flags. When StructuredAuthenticationConfiguration
	// feature gate is added and wired up, we will remove the 1 authenticator limit
	// check and add validation for duplicate issuers.
	for i, a := range c.JWT {
		fldPath := root.Index(i)
		allErrs = append(allErrs, validateJWTAuthenticator(a, fldPath)...)
	}

	return allErrs
}

// ValidateJWTAuthenticator validates a given JWTAuthenticator.
// This is exported for use in oidc package.
func ValidateJWTAuthenticator(authenticator api.JWTAuthenticator) field.ErrorList {
	return validateJWTAuthenticator(authenticator, nil)
}

func validateJWTAuthenticator(authenticator api.JWTAuthenticator, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList

	allErrs = append(allErrs, validateIssuer(authenticator.Issuer, fldPath.Child("issuer"))...)
	allErrs = append(allErrs, validateClaimValidationRules(authenticator.ClaimValidationRules, fldPath.Child("claimValidationRules"))...)
	allErrs = append(allErrs, validateClaimMappings(authenticator.ClaimMappings, fldPath.Child("claimMappings"))...)

	return allErrs
}

func validateIssuer(issuer api.Issuer, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList

	allErrs = append(allErrs, validateURL(issuer.URL, fldPath.Child("url"))...)
	allErrs = append(allErrs, validateAudiences(issuer.Audiences, fldPath.Child("audiences"))...)
	allErrs = append(allErrs, validateCertificateAuthority(issuer.CertificateAuthority, fldPath.Child("certificateAuthority"))...)

	return allErrs
}

func validateURL(issuerURL string, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList

	if len(issuerURL) == 0 {
		allErrs = append(allErrs, field.Required(fldPath, "URL is required"))
		return allErrs
	}

	u, err := url.Parse(issuerURL)
	if err != nil {
		allErrs = append(allErrs, field.Invalid(fldPath, issuerURL, err.Error()))
		return allErrs
	}
	if u.Scheme != "https" {
		allErrs = append(allErrs, field.Invalid(fldPath, issuerURL, "URL scheme must be https"))
	}
	if u.User != nil {
		allErrs = append(allErrs, field.Invalid(fldPath, issuerURL, "URL must not contain a username or password"))
	}
	if len(u.RawQuery) > 0 {
		allErrs = append(allErrs, field.Invalid(fldPath, issuerURL, "URL must not contain a query"))
	}
	if len(u.Fragment) > 0 {
		allErrs = append(allErrs, field.Invalid(fldPath, issuerURL, "URL must not contain a fragment"))
	}

	return allErrs
}

func validateAudiences(audiences []string, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList

	if len(audiences) == 0 {
		allErrs = append(allErrs, field.Required(fldPath, fmt.Sprintf(atLeastOneRequiredErrFmt, fldPath)))
		return allErrs
	}
	// This stricter validation is because the --oidc-client-id flag option is singular.
	// This will be removed when we support multiple audiences with the StructuredAuthenticationConfiguration feature gate.
	if len(audiences) > 1 {
		allErrs = append(allErrs, field.TooMany(fldPath, len(audiences), 1))
		return allErrs
	}

	for i, audience := range audiences {
		fldPath := fldPath.Index(i)
		if len(audience) == 0 {
			allErrs = append(allErrs, field.Required(fldPath, "audience can't be empty"))
		}
	}

	return allErrs
}

func validateCertificateAuthority(certificateAuthority string, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList

	if len(certificateAuthority) == 0 {
		return allErrs
	}
	_, err := cert.NewPoolFromBytes([]byte(certificateAuthority))
	if err != nil {
		allErrs = append(allErrs, field.Invalid(fldPath, "<omitted>", err.Error()))
	}

	return allErrs
}

func validateClaimValidationRules(rules []api.ClaimValidationRule, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList

	seenClaims := sets.NewString()
	for i, rule := range rules {
		fldPath := fldPath.Index(i)

		if len(rule.Claim) == 0 {
			allErrs = append(allErrs, field.Required(fldPath.Child("claim"), "claim name is required"))
			continue
		}

		if seenClaims.Has(rule.Claim) {
			allErrs = append(allErrs, field.Duplicate(fldPath.Child("claim"), rule.Claim))
			continue
		}
		seenClaims.Insert(rule.Claim)
	}

	return allErrs
}

func validateClaimMappings(m api.ClaimMappings, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList

	if len(m.Username.Claim) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("username", "claim"), "claim name is required"))
	}
	// TODO(aramase): when Expression is added to PrefixedClaimOrExpression, check prefix and expression are not both set.
	if m.Username.Prefix == nil {
		allErrs = append(allErrs, field.Required(fldPath.Child("username", "prefix"), "prefix is required"))
	}
	if len(m.Groups.Claim) > 0 && m.Groups.Prefix == nil {
		allErrs = append(allErrs, field.Required(fldPath.Child("groups", "prefix"), "prefix is required when claim is set"))
	}
	if m.Groups.Prefix != nil && len(m.Groups.Claim) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("groups", "claim"), "non-empty claim name is required when prefix is set"))
	}

	return allErrs
}
