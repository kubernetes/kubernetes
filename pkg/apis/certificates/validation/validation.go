/*
Copyright 2016 The Kubernetes Authors.

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

	utilvalidation "k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/apimachinery/pkg/util/validation/field"

	"k8s.io/kubernetes/pkg/apis/certificates"
	apivalidation "k8s.io/kubernetes/pkg/apis/core/validation"
)

// validateCSR validates the signature and formatting of a base64-wrapped,
// PEM-encoded PKCS#10 certificate signing request. If this is invalid, we must
// not accept the CSR for further processing.
func validateCSR(obj *certificates.CertificateSigningRequest) error {
	csr, err := certificates.ParseCSR(obj)
	if err != nil {
		return err
	}
	// check that the signature is valid
	err = csr.CheckSignature()
	if err != nil {
		return err
	}
	return nil
}

// We don't care what you call your certificate requests.
func ValidateCertificateRequestName(name string, prefix bool) []string {
	return nil
}

func ValidateCertificateSigningRequest(csr *certificates.CertificateSigningRequest) field.ErrorList {
	isNamespaced := false
	allErrs := apivalidation.ValidateObjectMeta(&csr.ObjectMeta, isNamespaced, ValidateCertificateRequestName, field.NewPath("metadata"))

	specPath := field.NewPath("spec")
	err := validateCSR(csr)
	if err != nil {
		allErrs = append(allErrs, field.Invalid(specPath.Child("request"), csr.Spec.Request, fmt.Sprintf("%v", err)))
	}
	if len(csr.Spec.Usages) == 0 {
		allErrs = append(allErrs, field.Required(specPath.Child("usages"), "usages must be provided"))
	}
	allErrs = append(allErrs, ValidateCertificateSigningRequestSignerName(specPath.Child("signerName"), csr.Spec.SignerName)...)
	return allErrs
}

// ensure signerName is of the form domain.com/something and up to 571 characters.
// This length and format is specified to accommodate signerNames like:
// <fqdn>/<resource-namespace>.<resource-name>.
// The max length of a FQDN is 253 characters (DNS1123Subdomain max length)
// The max length of a namespace name is 63 characters (DNS1123Label max length)
// The max length of a resource name is 253 characters (DNS1123Subdomain max length)
// We then add an additional 2 characters to account for the one '.' and one '/'.
func ValidateCertificateSigningRequestSignerName(fldPath *field.Path, signerName string) field.ErrorList {
	var el field.ErrorList
	if len(signerName) == 0 {
		el = append(el, field.Required(fldPath, "signerName must be provided"))
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
	maxDomainSegmentLength := utilvalidation.DNS1123SubdomainMaxLength
	if len(segments[0]) > maxDomainSegmentLength {
		el = append(el, field.TooLong(fldPath, segments[0], maxDomainSegmentLength))
	}
	// validate that segments[0] consists of valid DNS1123 labels separated by '.'
	domainLabels := strings.Split(segments[0], ".")
	for _, lbl := range domainLabels {
		// use IsDNS1123Label as we want to ensure the max length of any single label in the domain
		// is 63 characters
		if errs := utilvalidation.IsDNS1123Label(lbl); len(errs) > 0 {
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
		if errs := utilvalidation.IsDNS1123Subdomain(lbl); len(errs) > 0 {
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
	maxPathSegmentLength := utilvalidation.DNS1123SubdomainMaxLength + utilvalidation.DNS1123LabelMaxLength + 1
	maxSignerNameLength := maxDomainSegmentLength + maxPathSegmentLength + 1
	if len(signerName) > maxSignerNameLength {
		el = append(el, field.TooLong(fldPath, signerName, maxSignerNameLength))
	}

	return el
}

func ValidateCertificateSigningRequestUpdate(newCSR, oldCSR *certificates.CertificateSigningRequest) field.ErrorList {
	validationErrorList := ValidateCertificateSigningRequest(newCSR)
	metaUpdateErrorList := apivalidation.ValidateObjectMetaUpdate(&newCSR.ObjectMeta, &oldCSR.ObjectMeta, field.NewPath("metadata"))
	return append(validationErrorList, metaUpdateErrorList...)
}
