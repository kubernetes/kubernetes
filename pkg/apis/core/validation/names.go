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
	"strings"

	apimachineryvalidation "k8s.io/apimachinery/pkg/api/validation"
	"k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

// IsKubernetesSignerName checks if signerName is one reserved by the Kubernetes project.
func IsKubernetesSignerName(signerName string) bool {
	hostName, _, _ := strings.Cut(signerName, "/")
	return hostName == "kubernetes.io" || strings.HasSuffix(hostName, ".kubernetes.io")
}

// ValidateSignerName checks that signerName is syntactically valid.
//
// ensure signerName is of the form domain.com/something and up to 571 characters.
// This length and format is specified to accommodate signerNames like:
// <fqdn>/<resource-namespace>.<resource-name>.
// The max length of a FQDN is 253 characters (DNS1123Subdomain max length)
// The max length of a namespace name is 63 characters (DNS1123Label max length)
// The max length of a resource name is 253 characters (DNS1123Subdomain max length)
// We then add an additional 2 characters to account for the one '.' and one '/'.
func ValidateSignerName(fldPath *field.Path, signerName string) field.ErrorList {
	var el field.ErrorList
	if len(signerName) == 0 {
		el = append(el, field.Required(fldPath, ""))
		return el
	}

	segments := strings.Split(signerName, "/")
	// validate that there is one '/' in the signerName.
	// we do this after validating the domain segment to provide more info to the user.
	if len(segments) != 2 {
		el = append(el, field.Invalid(fldPath, signerName, "must be a fully qualified domain and path of the form 'example.com/signer-name'"))
		// return early here as we should not continue attempting to validate a missing or malformed path segment
		// (i.e. one containing multiple or zero `/`)
		return el
	}

	// validate that segments[0] is less than 253 characters altogether
	maxDomainSegmentLength := validation.DNS1123SubdomainMaxLength
	if len(segments[0]) > maxDomainSegmentLength {
		el = append(el, field.TooLong(fldPath, "" /*unused*/, maxDomainSegmentLength))
	}
	// validate that segments[0] consists of valid DNS1123 labels separated by '.'
	domainLabels := strings.Split(segments[0], ".")
	for _, lbl := range domainLabels {
		// use IsDNS1123Label as we want to ensure the max length of any single label in the domain
		// is 63 characters
		if errs := validation.IsDNS1123Label(lbl); len(errs) > 0 {
			for _, err := range errs {
				el = append(el, field.Invalid(fldPath, segments[0], fmt.Sprintf("validating label %q: %s", lbl, err)))
			}
			// if we encounter any errors whilst parsing the domain segment, break from
			// validation as any further error messages will be duplicates, and non-distinguishable
			// from each other, confusing users.
			break
		}
	}

	// validate that there is at least one '.' in segments[0]
	if len(domainLabels) < 2 {
		el = append(el, field.Invalid(fldPath, segments[0], "should be a domain with at least two segments separated by dots"))
	}

	// validate that segments[1] consists of valid DNS1123 subdomains separated by '.'.
	pathLabels := strings.Split(segments[1], ".")
	for _, lbl := range pathLabels {
		// use IsDNS1123Subdomain because it enforces a length restriction of 253 characters
		// which is required in order to fit a full resource name into a single 'label'
		if errs := validation.IsDNS1123Subdomain(lbl); len(errs) > 0 {
			for _, err := range errs {
				el = append(el, field.Invalid(fldPath, segments[1], fmt.Sprintf("validating label %q: %s", lbl, err)))
			}
			// if we encounter any errors whilst parsing the path segment, break from
			// validation as any further error messages will be duplicates, and non-distinguishable
			// from each other, confusing users.
			break
		}
	}

	// ensure that segments[1] can accommodate a dns label + dns subdomain + '.'
	maxPathSegmentLength := validation.DNS1123SubdomainMaxLength + validation.DNS1123LabelMaxLength + 1
	maxSignerNameLength := maxDomainSegmentLength + maxPathSegmentLength + 1
	if len(signerName) > maxSignerNameLength {
		el = append(el, field.TooLong(fldPath, "" /*unused*/, maxSignerNameLength))
	}

	return el
}

// ValidateClusterTrustBundleName checks that a ClusterTrustBundle name conforms
// to the rules documented on the type.
func ValidateClusterTrustBundleName(signerName string) func(name string, prefix bool) []string {
	return func(name string, isPrefix bool) []string {
		if signerName == "" {
			if strings.Contains(name, ":") {
				return []string{"ClusterTrustBundle without signer name must not have \":\" in its name"}
			}
			return apimachineryvalidation.NameIsDNSSubdomain(name, isPrefix)
		}

		requiredPrefix := strings.ReplaceAll(signerName, "/", ":") + ":"
		if !strings.HasPrefix(name, requiredPrefix) {
			return []string{fmt.Sprintf("ClusterTrustBundle for signerName %s must be named with prefix %s", signerName, requiredPrefix)}
		}
		return apimachineryvalidation.NameIsDNSSubdomain(strings.TrimPrefix(name, requiredPrefix), isPrefix)
	}
}

func extractSignerNameFromClusterTrustBundleName(name string) (string, bool) {
	if splitPoint := strings.LastIndex(name, ":"); splitPoint != -1 {
		// This looks like it refers to a signerName trustbundle.
		return strings.ReplaceAll(name[:splitPoint], ":", "/"), true
	} else {
		return "", false
	}
}
