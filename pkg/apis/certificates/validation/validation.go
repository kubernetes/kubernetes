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
	"bytes"
	"crypto/x509"
	"encoding/pem"
	"fmt"
	"strings"

	v1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/apimachinery/pkg/util/sets"
	utilvalidation "k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/apimachinery/pkg/util/validation/field"
	utilcert "k8s.io/client-go/util/cert"
	"k8s.io/kubernetes/pkg/apis/certificates"
	apivalidation "k8s.io/kubernetes/pkg/apis/core/validation"
)

var (
	// trueConditionTypes is the set of condition types which may only have a status of True if present
	trueConditionTypes = sets.NewString(
		string(certificates.CertificateApproved),
		string(certificates.CertificateDenied),
		string(certificates.CertificateFailed),
	)

	trueStatusOnly  = sets.NewString(string(v1.ConditionTrue))
	allStatusValues = sets.NewString(string(v1.ConditionTrue), string(v1.ConditionFalse), string(v1.ConditionUnknown))
)

type certificateValidationOptions struct {
	// The following allow modifications only permitted via certain update paths

	// allow populating/modifying Approved/Denied conditions
	allowSettingApprovalConditions bool
	// allow populating status.certificate
	allowSettingCertificate bool

	// allow Approved and Denied conditions to be exist.
	// we tolerate this when the problem is already present in the persisted object for compatibility.
	allowBothApprovedAndDenied bool

	// The following are bad things we tolerate for compatibility reasons:
	// * in requests made via the v1beta1 API
	// * in update requests where the problem is already present in the persisted object

	// allow modifying status.certificate on an update where the old object has a different certificate
	allowResettingCertificate bool
	// allow the legacy-unknown signerName
	allowLegacySignerName bool
	// allow conditions with duplicate types
	allowDuplicateConditionTypes bool
	// allow conditions with "" types
	allowEmptyConditionType bool
	// allow arbitrary content in status.certificate
	allowArbitraryCertificate bool
	// allow usages values outside the known set
	allowUnknownUsages bool
	// allow duplicate usages values
	allowDuplicateUsages bool
}

// validateCSR validates the signature and formatting of a base64-wrapped,
// PEM-encoded PKCS#10 certificate signing request. If this is invalid, we must
// not accept the CSR for further processing.
func validateCSR(obj *certificates.CertificateSigningRequest) error {
	csr, err := certificates.ParseCSR(obj.Spec.Request)
	if err != nil {
		return err
	}
	// check that the signature is valid
	return csr.CheckSignature()
}

func validateCertificate(pemData []byte) error {
	if len(pemData) == 0 {
		return nil
	}

	blocks := 0
	for {
		block, remainingData := pem.Decode(pemData)
		if block == nil {
			break
		}

		if block.Type != utilcert.CertificateBlockType {
			return fmt.Errorf("only CERTIFICATE PEM blocks are allowed, found %q", block.Type)
		}
		if len(block.Headers) != 0 {
			return fmt.Errorf("no PEM block headers are permitted")
		}
		blocks++

		certs, err := x509.ParseCertificates(block.Bytes)
		if err != nil {
			return err
		}
		if len(certs) == 0 {
			return fmt.Errorf("found CERTIFICATE PEM block containing 0 certificates")
		}

		pemData = remainingData
	}

	if blocks == 0 {
		return fmt.Errorf("must contain at least one CERTIFICATE PEM block")
	}

	return nil
}

// We don't care what you call your certificate requests.
func ValidateCertificateRequestName(name string, prefix bool) []string {
	return nil
}

func ValidateCertificateSigningRequestCreate(csr *certificates.CertificateSigningRequest) field.ErrorList {
	opts := getValidationOptions(csr, nil)
	return validateCertificateSigningRequest(csr, opts)
}

var (
	allValidUsages = sets.NewString(
		string(certificates.UsageSigning),
		string(certificates.UsageDigitalSignature),
		string(certificates.UsageContentCommitment),
		string(certificates.UsageKeyEncipherment),
		string(certificates.UsageKeyAgreement),
		string(certificates.UsageDataEncipherment),
		string(certificates.UsageCertSign),
		string(certificates.UsageCRLSign),
		string(certificates.UsageEncipherOnly),
		string(certificates.UsageDecipherOnly),
		string(certificates.UsageAny),
		string(certificates.UsageServerAuth),
		string(certificates.UsageClientAuth),
		string(certificates.UsageCodeSigning),
		string(certificates.UsageEmailProtection),
		string(certificates.UsageSMIME),
		string(certificates.UsageIPsecEndSystem),
		string(certificates.UsageIPsecTunnel),
		string(certificates.UsageIPsecUser),
		string(certificates.UsageTimestamping),
		string(certificates.UsageOCSPSigning),
		string(certificates.UsageMicrosoftSGC),
		string(certificates.UsageNetscapeSGC),
	)
)

func validateCertificateSigningRequest(csr *certificates.CertificateSigningRequest, opts certificateValidationOptions) field.ErrorList {
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
	if !opts.allowUnknownUsages {
		for i, usage := range csr.Spec.Usages {
			if !allValidUsages.Has(string(usage)) {
				allErrs = append(allErrs, field.NotSupported(specPath.Child("usages").Index(i), usage, allValidUsages.List()))
			}
		}
	}
	if !opts.allowDuplicateUsages {
		seen := make(map[certificates.KeyUsage]bool, len(csr.Spec.Usages))
		for i, usage := range csr.Spec.Usages {
			if seen[usage] {
				allErrs = append(allErrs, field.Duplicate(specPath.Child("usages").Index(i), usage))
			}
			seen[usage] = true
		}
	}
	if !opts.allowLegacySignerName && csr.Spec.SignerName == certificates.LegacyUnknownSignerName {
		allErrs = append(allErrs, field.Invalid(specPath.Child("signerName"), csr.Spec.SignerName, "the legacy signerName is not allowed via this API version"))
	} else {
		allErrs = append(allErrs, ValidateCertificateSigningRequestSignerName(specPath.Child("signerName"), csr.Spec.SignerName)...)
	}
	if csr.Spec.ExpirationSeconds != nil && *csr.Spec.ExpirationSeconds < 600 {
		allErrs = append(allErrs, field.Invalid(specPath.Child("expirationSeconds"), *csr.Spec.ExpirationSeconds, "may not specify a duration less than 600 seconds (10 minutes)"))
	}
	allErrs = append(allErrs, validateConditions(field.NewPath("status", "conditions"), csr, opts)...)

	if !opts.allowArbitraryCertificate {
		if err := validateCertificate(csr.Status.Certificate); err != nil {
			allErrs = append(allErrs, field.Invalid(field.NewPath("status", "certificate"), "<certificate data>", err.Error()))
		}
	}

	return allErrs
}

func validateConditions(fldPath *field.Path, csr *certificates.CertificateSigningRequest, opts certificateValidationOptions) field.ErrorList {
	allErrs := field.ErrorList{}

	seenTypes := map[certificates.RequestConditionType]bool{}
	hasApproved := false
	hasDenied := false

	for i, c := range csr.Status.Conditions {

		if !opts.allowEmptyConditionType {
			if len(c.Type) == 0 {
				allErrs = append(allErrs, field.Required(fldPath.Index(i).Child("type"), ""))
			}
		}

		allowedStatusValues := allStatusValues
		if trueConditionTypes.Has(string(c.Type)) {
			allowedStatusValues = trueStatusOnly
		}
		switch {
		case c.Status == "":
			allErrs = append(allErrs, field.Required(fldPath.Index(i).Child("status"), ""))
		case !allowedStatusValues.Has(string(c.Status)):
			allErrs = append(allErrs, field.NotSupported(fldPath.Index(i).Child("status"), c.Status, allowedStatusValues.List()))
		}

		if !opts.allowBothApprovedAndDenied {
			switch c.Type {
			case certificates.CertificateApproved:
				hasApproved = true
				if hasDenied {
					allErrs = append(allErrs, field.Invalid(fldPath.Index(i).Child("type"), c.Type, "Approved and Denied conditions are mutually exclusive"))
				}
			case certificates.CertificateDenied:
				hasDenied = true
				if hasApproved {
					allErrs = append(allErrs, field.Invalid(fldPath.Index(i).Child("type"), c.Type, "Approved and Denied conditions are mutually exclusive"))
				}
			}
		}

		if !opts.allowDuplicateConditionTypes {
			if seenTypes[c.Type] {
				allErrs = append(allErrs, field.Duplicate(fldPath.Index(i).Child("type"), c.Type))
			}
			seenTypes[c.Type] = true
		}
	}

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
	opts := getValidationOptions(newCSR, oldCSR)
	return validateCertificateSigningRequestUpdate(newCSR, oldCSR, opts)
}

func ValidateCertificateSigningRequestStatusUpdate(newCSR, oldCSR *certificates.CertificateSigningRequest) field.ErrorList {
	opts := getValidationOptions(newCSR, oldCSR)
	opts.allowSettingCertificate = true
	return validateCertificateSigningRequestUpdate(newCSR, oldCSR, opts)
}

func ValidateCertificateSigningRequestApprovalUpdate(newCSR, oldCSR *certificates.CertificateSigningRequest) field.ErrorList {
	opts := getValidationOptions(newCSR, oldCSR)
	opts.allowSettingApprovalConditions = true
	return validateCertificateSigningRequestUpdate(newCSR, oldCSR, opts)
}

func validateCertificateSigningRequestUpdate(newCSR, oldCSR *certificates.CertificateSigningRequest, opts certificateValidationOptions) field.ErrorList {
	validationErrorList := validateCertificateSigningRequest(newCSR, opts)
	metaUpdateErrorList := apivalidation.ValidateObjectMetaUpdate(&newCSR.ObjectMeta, &oldCSR.ObjectMeta, field.NewPath("metadata"))

	// prevent removal of existing Approved/Denied/Failed conditions
	for _, t := range []certificates.RequestConditionType{certificates.CertificateApproved, certificates.CertificateDenied, certificates.CertificateFailed} {
		oldConditions := findConditions(oldCSR, t)
		newConditions := findConditions(newCSR, t)
		if len(newConditions) < len(oldConditions) {
			validationErrorList = append(validationErrorList, field.Forbidden(field.NewPath("status", "conditions"), fmt.Sprintf("updates may not remove a condition of type %q", t)))
		}
	}

	if !opts.allowSettingApprovalConditions {
		// prevent addition/removal/modification of Approved/Denied conditions
		for _, t := range []certificates.RequestConditionType{certificates.CertificateApproved, certificates.CertificateDenied} {
			oldConditions := findConditions(oldCSR, t)
			newConditions := findConditions(newCSR, t)
			switch {
			case len(newConditions) < len(oldConditions):
				// removals are prevented above
			case len(newConditions) > len(oldConditions):
				validationErrorList = append(validationErrorList, field.Forbidden(field.NewPath("status", "conditions"), fmt.Sprintf("updates may not add a condition of type %q", t)))
			case !apiequality.Semantic.DeepEqual(oldConditions, newConditions):
				conditionDiff := diff.ObjectDiff(oldConditions, newConditions)
				validationErrorList = append(validationErrorList, field.Forbidden(field.NewPath("status", "conditions"), fmt.Sprintf("updates may not modify a condition of type %q\n%v", t, conditionDiff)))
			}
		}
	}

	if !bytes.Equal(newCSR.Status.Certificate, oldCSR.Status.Certificate) {
		if !opts.allowSettingCertificate {
			validationErrorList = append(validationErrorList, field.Forbidden(field.NewPath("status", "certificate"), "updates may not set certificate content"))
		} else if !opts.allowResettingCertificate && len(oldCSR.Status.Certificate) > 0 {
			validationErrorList = append(validationErrorList, field.Forbidden(field.NewPath("status", "certificate"), "updates may not modify existing certificate content"))
		}
	}

	return append(validationErrorList, metaUpdateErrorList...)
}

// findConditions returns all instances of conditions of the specified type
func findConditions(csr *certificates.CertificateSigningRequest, conditionType certificates.RequestConditionType) []certificates.CertificateSigningRequestCondition {
	var retval []certificates.CertificateSigningRequestCondition
	for i, c := range csr.Status.Conditions {
		if c.Type == conditionType {
			retval = append(retval, csr.Status.Conditions[i])
		}
	}
	return retval
}

// getValidationOptions returns the validation options to be
// compatible with the specified version and existing CSR.
// oldCSR may be nil if this is a create request.
// validation options related to subresource-specific capabilities are set to false.
func getValidationOptions(newCSR, oldCSR *certificates.CertificateSigningRequest) certificateValidationOptions {
	return certificateValidationOptions{
		allowResettingCertificate:    false,
		allowBothApprovedAndDenied:   allowBothApprovedAndDenied(oldCSR),
		allowLegacySignerName:        allowLegacySignerName(oldCSR),
		allowDuplicateConditionTypes: allowDuplicateConditionTypes(oldCSR),
		allowEmptyConditionType:      allowEmptyConditionType(oldCSR),
		allowArbitraryCertificate:    allowArbitraryCertificate(newCSR, oldCSR),
		allowDuplicateUsages:         allowDuplicateUsages(oldCSR),
		allowUnknownUsages:           allowUnknownUsages(oldCSR),
	}
}

func allowBothApprovedAndDenied(oldCSR *certificates.CertificateSigningRequest) bool {
	if oldCSR == nil {
		return false
	}
	approved := false
	denied := false
	for _, c := range oldCSR.Status.Conditions {
		if c.Type == certificates.CertificateApproved {
			approved = true
		} else if c.Type == certificates.CertificateDenied {
			denied = true
		}
	}
	// compatibility with existing data
	return approved && denied
}

func allowLegacySignerName(oldCSR *certificates.CertificateSigningRequest) bool {
	switch {
	case oldCSR != nil && oldCSR.Spec.SignerName == certificates.LegacyUnknownSignerName:
		return true // compatibility with existing data
	default:
		return false
	}
}

func allowDuplicateConditionTypes(oldCSR *certificates.CertificateSigningRequest) bool {
	switch {
	case oldCSR != nil && hasDuplicateConditionTypes(oldCSR):
		return true // compatibility with existing data
	default:
		return false
	}
}
func hasDuplicateConditionTypes(csr *certificates.CertificateSigningRequest) bool {
	seen := map[certificates.RequestConditionType]bool{}
	for _, c := range csr.Status.Conditions {
		if seen[c.Type] {
			return true
		}
		seen[c.Type] = true
	}
	return false
}

func allowEmptyConditionType(oldCSR *certificates.CertificateSigningRequest) bool {
	switch {
	case oldCSR != nil && hasEmptyConditionType(oldCSR):
		return true // compatibility with existing data
	default:
		return false
	}
}
func hasEmptyConditionType(csr *certificates.CertificateSigningRequest) bool {
	for _, c := range csr.Status.Conditions {
		if len(c.Type) == 0 {
			return true
		}
	}
	return false
}

func allowArbitraryCertificate(newCSR, oldCSR *certificates.CertificateSigningRequest) bool {
	switch {
	case newCSR != nil && oldCSR != nil && bytes.Equal(newCSR.Status.Certificate, oldCSR.Status.Certificate):
		return true // tolerate updates that don't touch status.certificate
	case oldCSR != nil && validateCertificate(oldCSR.Status.Certificate) != nil:
		return true // compatibility with existing data
	default:
		return false
	}
}

func allowUnknownUsages(oldCSR *certificates.CertificateSigningRequest) bool {
	switch {
	case oldCSR != nil && hasUnknownUsage(oldCSR.Spec.Usages):
		return true // compatibility with existing data
	default:
		return false
	}
}

func hasUnknownUsage(usages []certificates.KeyUsage) bool {
	for _, usage := range usages {
		if !allValidUsages.Has(string(usage)) {
			return true
		}
	}
	return false
}

func allowDuplicateUsages(oldCSR *certificates.CertificateSigningRequest) bool {
	switch {
	case oldCSR != nil && hasDuplicateUsage(oldCSR.Spec.Usages):
		return true // compatibility with existing data
	default:
		return false
	}
}

func hasDuplicateUsage(usages []certificates.KeyUsage) bool {
	seen := make(map[certificates.KeyUsage]bool, len(usages))
	for _, usage := range usages {
		if seen[usage] {
			return true
		}
		seen[usage] = true
	}
	return false
}
