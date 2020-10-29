package oauth

import (
	"strings"

	"k8s.io/apimachinery/pkg/util/validation/field"

	configv1 "github.com/openshift/api/config/v1"
	crvalidation "k8s.io/kubernetes/openshift-kube-apiserver/admission/customresourcevalidation"
)

func ValidateGitHubIdentityProvider(provider *configv1.GitHubIdentityProvider, mappingMethod configv1.MappingMethodType, fieldPath *field.Path) field.ErrorList {
	errs := field.ErrorList{}
	if provider == nil {
		errs = append(errs, field.Required(fieldPath, ""))
		return errs
	}

	errs = append(errs, ValidateOAuthIdentityProvider(provider.ClientID, provider.ClientSecret, fieldPath.Child("provider"))...)

	if len(provider.Teams) > 0 && len(provider.Organizations) > 0 {
		errs = append(errs, field.Invalid(fieldPath.Child("organizations"), provider.Organizations, "specify organizations or teams, not both"))
		errs = append(errs, field.Invalid(fieldPath.Child("teams"), provider.Teams, "specify organizations or teams, not both"))
	}

	// only check that there are some teams/orgs if not GitHub Enterprise Server
	if len(provider.Hostname) == 0 && len(provider.Teams) == 0 && len(provider.Organizations) == 0 && mappingMethod != configv1.MappingMethodLookup {
		errs = append(errs, field.Invalid(fieldPath, nil, "one of organizations or teams must be specified unless hostname is set or lookup is used"))
	}
	for i, organization := range provider.Organizations {
		if strings.Contains(organization, "/") {
			errs = append(errs, field.Invalid(fieldPath.Child("organizations").Index(i), organization, "cannot contain /"))
		}
		if len(organization) == 0 {
			errs = append(errs, field.Required(fieldPath.Child("organizations").Index(i), "cannot be empty"))
		}
	}
	for i, team := range provider.Teams {
		if split := strings.Split(team, "/"); len(split) != 2 {
			errs = append(errs, field.Invalid(fieldPath.Child("teams").Index(i), team, "must be in the format <org>/<team>"))
		} else if org, t := split[0], split[1]; len(org) == 0 || len(t) == 0 {
			errs = append(errs, field.Invalid(fieldPath.Child("teams").Index(i), team, "must be in the format <org>/<team>"))
		}
	}

	if hostname := provider.Hostname; len(hostname) != 0 {
		hostnamePath := fieldPath.Child("hostname")

		if hostname == "github.com" || strings.HasSuffix(hostname, ".github.com") {
			errs = append(errs, field.Invalid(hostnamePath, hostname, "cannot equal [*.]github.com"))
		}

		if !isValidHostname(hostname) {
			errs = append(errs, field.Invalid(hostnamePath, hostname, "must be a valid DNS subdomain or IP address"))
		}
	}

	if caFile := provider.CA; len(caFile.Name) != 0 {
		caPath := fieldPath.Child("ca")

		errs = append(errs, crvalidation.ValidateConfigMapReference(caPath, caFile, true)...)

		if len(provider.Hostname) == 0 {
			errs = append(errs, field.Invalid(caPath, caFile, "cannot be specified when hostname is empty"))
		}
	}

	return errs
}
