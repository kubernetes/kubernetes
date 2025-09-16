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
	"crypto"
	"crypto/ecdsa"
	"crypto/ed25519"
	"crypto/elliptic"
	"crypto/rsa"
	"crypto/sha256"
	"crypto/x509"
	"encoding/pem"
	"fmt"
	"net/mail"
	"strconv"
	"strings"
	"time"

	v1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	apimachineryvalidation "k8s.io/apimachinery/pkg/api/validation"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	metav1validation "k8s.io/apimachinery/pkg/apis/meta/v1/validation"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation/field"
	utilcert "k8s.io/client-go/util/cert"
	"k8s.io/kubernetes/pkg/apis/certificates"
	apivalidation "k8s.io/kubernetes/pkg/apis/core/validation"
	"k8s.io/utils/clock"
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
		allErrs = append(allErrs, field.Required(specPath.Child("usages"), ""))
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
		allErrs = append(allErrs, apivalidation.ValidateSignerName(specPath.Child("signerName"), csr.Spec.SignerName)...)
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
					allErrs = append(allErrs, field.Invalid(fldPath, c.Type, "Approved and Denied conditions are mutually exclusive").WithOrigin("zeroOrOneOf").MarkCoveredByDeclarative())
				}
			case certificates.CertificateDenied:
				hasDenied = true
				if hasApproved {
					allErrs = append(allErrs, field.Invalid(fldPath, c.Type, "Approved and Denied conditions are mutually exclusive").WithOrigin("zeroOrOneOf").MarkCoveredByDeclarative())
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
				conditionDiff := diff.Diff(oldConditions, newConditions)
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

type ValidateClusterTrustBundleOptions struct {
	SuppressBundleParsing bool
}

// ValidateClusterTrustBundle runs all validation checks on bundle.
func ValidateClusterTrustBundle(bundle *certificates.ClusterTrustBundle, opts ValidateClusterTrustBundleOptions) field.ErrorList {
	var allErrors field.ErrorList

	metaErrors := apivalidation.ValidateObjectMeta(&bundle.ObjectMeta, false, apivalidation.ValidateClusterTrustBundleName(bundle.Spec.SignerName), field.NewPath("metadata"))
	allErrors = append(allErrors, metaErrors...)

	if bundle.Spec.SignerName != "" {
		signerNameErrors := apivalidation.ValidateSignerName(field.NewPath("spec", "signerName"), bundle.Spec.SignerName)
		allErrors = append(allErrors, signerNameErrors...)
	}

	if !opts.SuppressBundleParsing {
		pemErrors := validateTrustBundle(field.NewPath("spec", "trustBundle"), bundle.Spec.TrustBundle)
		allErrors = append(allErrors, pemErrors...)
	}

	return allErrors
}

// ValidateClusterTrustBundleUpdate runs all update validation checks on an
// update.
func ValidateClusterTrustBundleUpdate(newBundle, oldBundle *certificates.ClusterTrustBundle) field.ErrorList {
	// If the caller isn't changing the TrustBundle field, don't parse it.
	// This helps smoothly handle changes in Go's PEM or X.509 parsing
	// libraries.
	opts := ValidateClusterTrustBundleOptions{}
	if newBundle.Spec.TrustBundle == oldBundle.Spec.TrustBundle {
		opts.SuppressBundleParsing = true
	}

	var allErrors field.ErrorList
	allErrors = append(allErrors, ValidateClusterTrustBundle(newBundle, opts)...)
	allErrors = append(allErrors, apivalidation.ValidateObjectMetaUpdate(&newBundle.ObjectMeta, &oldBundle.ObjectMeta, field.NewPath("metadata"))...)
	allErrors = append(allErrors, apivalidation.ValidateImmutableField(newBundle.Spec.SignerName, oldBundle.Spec.SignerName, field.NewPath("spec", "signerName"))...)
	return allErrors
}

// validateTrustBundle rejects intra-block headers, blocks
// that don't parse as X.509 CA certificates, and duplicate trust anchors.  It
// requires that at least one trust anchor is provided.
func validateTrustBundle(path *field.Path, in string) field.ErrorList {
	var allErrors field.ErrorList

	if len(in) > certificates.MaxTrustBundleSize {
		allErrors = append(allErrors, field.TooLong(path, "" /*unused*/, certificates.MaxTrustBundleSize))
		return allErrors
	}

	blockDedupe := map[string][]int{}

	rest := []byte(in)
	var b *pem.Block
	i := -1
	for {
		b, rest = pem.Decode(rest)
		if b == nil {
			break
		}
		i++

		if b.Type != "CERTIFICATE" {
			allErrors = append(allErrors, field.Invalid(path, "<value omitted>", fmt.Sprintf("entry %d has bad block type: %v", i, b.Type)))
			continue
		}

		if len(b.Headers) != 0 {
			allErrors = append(allErrors, field.Invalid(path, "<value omitted>", fmt.Sprintf("entry %d has PEM block headers", i)))
			continue
		}

		cert, err := x509.ParseCertificate(b.Bytes)
		if err != nil {
			allErrors = append(allErrors, field.Invalid(path, "<value omitted>", fmt.Sprintf("entry %d does not parse as X.509", i)))
			continue
		}

		if !cert.IsCA {
			allErrors = append(allErrors, field.Invalid(path, "<value omitted>", fmt.Sprintf("entry %d does not have the CA bit set", i)))
			continue
		}

		if !cert.BasicConstraintsValid {
			allErrors = append(allErrors, field.Invalid(path, "<value omitted>", fmt.Sprintf("entry %d has invalid basic constraints", i)))
			continue
		}

		blockDedupe[string(b.Bytes)] = append(blockDedupe[string(b.Bytes)], i)
	}

	// If we had a malformed block, don't also output potentially-redundant
	// errors about duplicate or missing trust anchors.
	if len(allErrors) != 0 {
		return allErrors
	}

	if len(blockDedupe) == 0 {
		allErrors = append(allErrors, field.Invalid(path, "<value omitted>", "at least one trust anchor must be provided"))
	}

	for _, indices := range blockDedupe {
		if len(indices) > 1 {
			allErrors = append(allErrors, field.Invalid(path, "<value omitted>", fmt.Sprintf("duplicate trust anchor (indices %v)", indices)))
		}
	}

	return allErrors
}

// ValidatePodCertificateRequestCreate runs all validation checks on a pod certificate request create.
func ValidatePodCertificateRequestCreate(req *certificates.PodCertificateRequest) field.ErrorList {
	var allErrors field.ErrorList

	metaErrors := apivalidation.ValidateObjectMeta(&req.ObjectMeta, true, apimachineryvalidation.NameIsDNSSubdomain, field.NewPath("metadata"))
	allErrors = append(allErrors, metaErrors...)

	signerNameErrors := apivalidation.ValidateSignerName(field.NewPath("spec", "signerName"), req.Spec.SignerName)
	allErrors = append(allErrors, signerNameErrors...)

	if req.Spec.UnverifiedUserAnnotations != nil {
		userAnnotationsErrors := apivalidation.ValidateUserAnnotations(req.Spec.UnverifiedUserAnnotations, field.NewPath("spec", "unverifiedUserAnnotations"))
		allErrors = append(allErrors, userAnnotationsErrors...)
	}

	for _, msg := range apivalidation.ValidatePodName(req.Spec.PodName, false) {
		allErrors = append(allErrors, field.Invalid(field.NewPath("spec", "podName"), req.Spec.PodName, msg))
	}
	if len(req.Spec.PodUID) == 0 {
		allErrors = append(allErrors, field.Invalid(field.NewPath("spec", "podUID"), req.Spec.PodUID, "must not be empty"))
	}
	if len(req.Spec.PodUID) > 128 {
		allErrors = append(allErrors, field.TooLong(field.NewPath("spec", "podUID"), req.Spec.PodUID, 128))
	}
	for _, msg := range apivalidation.ValidateServiceAccountName(req.Spec.ServiceAccountName, false) {
		allErrors = append(allErrors, field.Invalid(field.NewPath("spec", "serviceAccountName"), req.Spec.ServiceAccountName, msg))
	}
	if len(req.Spec.ServiceAccountUID) == 0 {
		allErrors = append(allErrors, field.Invalid(field.NewPath("spec", "serviceAccountUID"), req.Spec.ServiceAccountUID, "must not be empty"))
	}
	if len(req.Spec.ServiceAccountUID) > 128 {
		allErrors = append(allErrors, field.TooLong(field.NewPath("spec", "serviceAccountUID"), req.Spec.ServiceAccountUID, 128))
	}
	for _, msg := range apivalidation.ValidateNodeName(string(req.Spec.NodeName), false) {
		allErrors = append(allErrors, field.Invalid(field.NewPath("spec", "nodeName"), req.Spec.NodeName, msg))
	}
	if len(req.Spec.NodeUID) == 0 {
		allErrors = append(allErrors, field.Invalid(field.NewPath("spec", "nodeUID"), req.Spec.NodeUID, "must not be empty"))
	}
	if len(req.Spec.NodeUID) > 128 {
		allErrors = append(allErrors, field.TooLong(field.NewPath("spec", "nodeUID"), req.Spec.NodeUID, 128))
	}

	if req.Spec.MaxExpirationSeconds == nil {
		allErrors = append(allErrors, field.Required(field.NewPath("spec", "maxExpirationSeconds"), "must be set"))
		return allErrors
	}
	if apivalidation.IsKubernetesSignerName(req.Spec.SignerName) {
		// Kubernetes signers are restricted to max 24 hour certs
		if !(certificates.MinMaxExpirationSeconds <= *req.Spec.MaxExpirationSeconds && *req.Spec.MaxExpirationSeconds <= certificates.KubernetesMaxMaxExpirationSeconds) {
			allErrors = append(allErrors, field.Invalid(field.NewPath("spec", "maxExpirationSeconds"), req.Spec.MaxExpirationSeconds, fmt.Sprintf("must be in the range [%d, %d]", certificates.MinMaxExpirationSeconds, certificates.KubernetesMaxMaxExpirationSeconds)))
		}
	} else {
		// All other signers are restricted to max 91 day certs.
		if !(certificates.MinMaxExpirationSeconds <= *req.Spec.MaxExpirationSeconds && *req.Spec.MaxExpirationSeconds <= certificates.MaxMaxExpirationSeconds) {
			allErrors = append(allErrors, field.Invalid(field.NewPath("spec", "maxExpirationSeconds"), req.Spec.MaxExpirationSeconds, fmt.Sprintf("must be in the range [%d, %d]", certificates.MinMaxExpirationSeconds, certificates.MaxMaxExpirationSeconds)))
		}
	}

	if len(req.Spec.PKIXPublicKey) > certificates.MaxPKIXPublicKeySize {
		allErrors = append(allErrors, field.TooLong(field.NewPath("spec", "pkixPublicKey"), req.Spec.PKIXPublicKey, certificates.MaxPKIXPublicKeySize))
		return allErrors
	}

	if len(req.Spec.ProofOfPossession) > certificates.MaxProofOfPossessionSize {
		allErrors = append(allErrors, field.TooLong(field.NewPath("spec", "proofOfPossession"), req.Spec.ProofOfPossession, certificates.MaxProofOfPossessionSize))
		return allErrors
	}

	pubAny, err := x509.ParsePKIXPublicKey(req.Spec.PKIXPublicKey)
	if err != nil {
		allErrors = append(allErrors, field.Invalid(pkixPath, req.Spec.PKIXPublicKey, "must be a valid PKIX-serialized public key"))
		return allErrors
	}

	// Verify public key properties and the proof-of-possession signature.
	switch pub := pubAny.(type) {
	case ed25519.PublicKey:
		// ed25519 has no key configuration to check
		if !ed25519.Verify(pub, []byte(req.Spec.PodUID), req.Spec.ProofOfPossession) {
			allErrors = append(allErrors, field.Invalid(popPath, field.OmitValueType{}, "could not verify proof-of-possession signature"))
			return allErrors
		}

	case *ecdsa.PublicKey:
		if pub.Curve != elliptic.P256() && pub.Curve != elliptic.P384() && pub.Curve != elliptic.P521() {
			allErrors = append(allErrors, field.Invalid(pkixPath, "curve "+pub.Curve.Params().Name, "elliptic public keys must use curve P256 or P384"))
			return allErrors
		}
		if !ecdsa.VerifyASN1(pub, hashBytes([]byte(req.Spec.PodUID)), req.Spec.ProofOfPossession) {
			allErrors = append(allErrors, field.Invalid(popPath, field.OmitValueType{}, "could not verify proof-of-possession signature"))
			return allErrors
		}

	case *rsa.PublicKey:
		if pub.Size()*8 != 3072 && pub.Size()*8 != 4096 {
			allErrors = append(allErrors, field.Invalid(pkixPath, fmt.Sprintf("%d-bit modulus", pub.Size()*8), "RSA keys must have modulus size 3072 or 4096"))
			return allErrors
		}
		if err := rsa.VerifyPSS(pub, crypto.SHA256, hashBytes([]byte(req.Spec.PodUID)), req.Spec.ProofOfPossession, nil); err != nil {
			allErrors = append(allErrors, field.Invalid(popPath, field.OmitValueType{}, "could not verify proof-of-possession signature"))
			return allErrors
		}

	default:
		allErrors = append(allErrors, field.Invalid(pkixPath, req.Spec.PKIXPublicKey, "unknown public key type; supported types are Ed25519, ECDSA, and RSA"))
		return allErrors
	}

	return allErrors
}

func hashBytes(in []byte) []byte {
	out := sha256.Sum256(in)
	return out[:]
}

var (
	pkixPath         = field.NewPath("spec", "pkixPublicKey")
	popPath          = field.NewPath("spec", "proofOfPossession")
	certChainPath    = field.NewPath("status", "certificateChain")
	notBeforePath    = field.NewPath("status", "notBefore")
	notAfterPath     = field.NewPath("status", "notAfter")
	beginRefreshPath = field.NewPath("status", "beginRefreshAt")
)

// ValidatePodCertificateRequestUpdate runs all update validation checks on a
// non-status update.
//
// All spec fields are immutable after creation, and status updates must go
// through the dedicated status update verb, so only metadata updates are
// allowed.
func ValidatePodCertificateRequestUpdate(newReq, oldReq *certificates.PodCertificateRequest) field.ErrorList {
	var allErrors field.ErrorList
	allErrors = append(allErrors, apivalidation.ValidateObjectMetaUpdate(&newReq.ObjectMeta, &oldReq.ObjectMeta, field.NewPath("metadata"))...)

	// All spec fields are immutable.
	allErrors = append(allErrors, apivalidation.ValidateImmutableField(newReq.Spec, oldReq.Spec, field.NewPath("spec"))...)

	return allErrors
}

// ValidatePodCertificateRequestStatusUpdate validates a status update for a
// PodCertificateRequest.
func ValidatePodCertificateRequestStatusUpdate(newReq, oldReq *certificates.PodCertificateRequest, clock clock.PassiveClock) field.ErrorList {
	var allErrors field.ErrorList

	// Metadata is *mostly* immutable... ManagedFields is allowed to change.  We
	// are reliant on the strategy that's calling us to have patched
	// newReq.ObjectMeta using metav1.ResetObjectMetaForStatus.
	allErrors = append(allErrors, apivalidation.ValidateObjectMetaUpdate(&newReq.ObjectMeta, &oldReq.ObjectMeta, field.NewPath("metadata"))...)
	if len(allErrors) > 0 {
		return allErrors
	}

	// Don't validate spec.  Strategy has stomped it.

	// There can be at most one of the known conditions, and it must have status "True"
	numKnownConditions := 0
	for i, cond := range newReq.Status.Conditions {
		switch cond.Type {
		case certificates.PodCertificateRequestConditionTypeIssued, certificates.PodCertificateRequestConditionTypeDenied, certificates.PodCertificateRequestConditionTypeFailed:
			numKnownConditions++
			if numKnownConditions > 1 {
				allErrors = append(allErrors, field.Invalid(field.NewPath("status", "conditions", formatIndex(i), "type"), cond.Type, `There may be at most one condition with type "Issued", "Denied", or "Failed"`))
			}
			if cond.Status != metav1.ConditionTrue {
				allErrors = append(allErrors, field.NotSupported(field.NewPath("status", "conditions", formatIndex(i), "status"), cond.Status, []metav1.ConditionStatus{metav1.ConditionTrue}))
			}
		default:
			allErrors = append(allErrors, field.NotSupported(field.NewPath("status", "conditions", formatIndex(i), "type"), cond.Type, []string{certificates.PodCertificateRequestConditionTypeIssued, certificates.PodCertificateRequestConditionTypeDenied, certificates.PodCertificateRequestConditionTypeFailed}))
		}
	}

	allErrors = append(allErrors, metav1validation.ValidateConditions(newReq.Status.Conditions, field.NewPath("status", "conditions"))...)

	// Bail if something seems wrong with the conditions --- we use the
	// conditions to drive validation of the remainder of the status fields.
	if len(allErrors) > 0 {
		return allErrors
	}

	// Is the original PCR in a terminal condition?  If so, the entire status
	// field (including conditions) is immutable.  No more changes are
	// permitted.
	if pcrIsIssued(oldReq) || pcrIsDenied(oldReq) || pcrIsFailed(oldReq) {
		allErrors = append(allErrors, validateSemanticEquality(newReq.Status, oldReq.Status, field.NewPath("status"), "immutable after PodCertificateRequest is issued, denied, or failed")...)
		return allErrors
	}

	// Are we transitioning to the "denied" or "failed" terminal conditions?
	if pcrIsDenied(newReq) || pcrIsFailed(newReq) {
		// No other status fields may change besides conditions.
		wantStatus := certificates.PodCertificateRequestStatus{
			Conditions: newReq.Status.Conditions,
		}
		allErrors = append(allErrors, validateSemanticEquality(newReq.Status, wantStatus, field.NewPath("status"), "non-condition status fields must be empty when denying or failing the PodCertificateRequest")...)
		return allErrors
	}

	// Are we transitioning to the "issued" terminal condition?
	if pcrIsIssued(newReq) {
		if len(newReq.Status.CertificateChain) > certificates.MaxCertificateChainSize {
			allErrors = append(allErrors, field.TooLong(field.NewPath("status", "certificateChain"), newReq.Status.CertificateChain, certificates.MaxCertificateChainSize))
			return allErrors
		}

		leafBlock, rest := pem.Decode([]byte(newReq.Status.CertificateChain))
		if leafBlock == nil {
			allErrors = append(allErrors, field.Invalid(certChainPath, newReq.Status.CertificateChain, "issued certificate chain must contain at least one certificate"))
			return allErrors
		}
		if leafBlock.Type != "CERTIFICATE" {
			allErrors = append(allErrors, field.Invalid(certChainPath, newReq.Status.CertificateChain, "issued certificate chain must consist entirely of CERTIFICATE PEM blocks"))
			return allErrors
		}

		leafCert, err := x509.ParseCertificate(leafBlock.Bytes)
		if err != nil {
			allErrors = append(allErrors, field.Invalid(certChainPath, newReq.Status.CertificateChain, "leaf certificate does not parse as valid X.509"))
			return allErrors
		}
		for _, dnsName := range leafCert.DNSNames {
			if dnsName == "" {
				allErrors = append(allErrors, field.Invalid(certChainPath, dnsName, "leaf certificate should not contain empty DNSName"))
			}
			if strings.Contains(dnsName, "..") {
				allErrors = append(allErrors, field.Invalid(certChainPath, dnsName, "leaf certificate's DNSName should not contain '..'"))
			}
			if strings.HasPrefix(dnsName, ".") || strings.HasSuffix(dnsName, ".") {
				allErrors = append(allErrors, field.Invalid(certChainPath, dnsName, "leaf certificate's DNSName should not start or end with '.'"))
			}
		}
		for _, emailAddress := range leafCert.EmailAddresses {
			if _, err := mail.ParseAddress(emailAddress); err != nil {
				allErrors = append(allErrors, field.Invalid(certChainPath, emailAddress, "leaf certificate should not contain invalid EmailAddress"))
			}
		}

		// Was the certificate issued to the public key in the spec?
		wantPKAny, err := x509.ParsePKIXPublicKey(oldReq.Spec.PKIXPublicKey)
		if err != nil {
			allErrors = append(allErrors, field.Invalid(pkixPath, oldReq.Spec.PKIXPublicKey, "must be a valid PKIX-serialized public key"))
			return allErrors
		}
		switch wantPK := wantPKAny.(type) {
		case ed25519.PublicKey:
			if !wantPK.Equal(leafCert.PublicKey) {
				allErrors = append(allErrors, field.Invalid(certChainPath, newReq.Status.CertificateChain, "leaf certificate was not issued to the requested public key"))
				return allErrors
			}
		case *rsa.PublicKey:
			if !wantPK.Equal(leafCert.PublicKey) {
				allErrors = append(allErrors, field.Invalid(certChainPath, newReq.Status.CertificateChain, "leaf certificate was not issued to the requested public key"))
				return allErrors
			}
		case *ecdsa.PublicKey:
			if !wantPK.Equal(leafCert.PublicKey) {
				allErrors = append(allErrors, field.Invalid(certChainPath, newReq.Status.CertificateChain, "leaf certificate was not issued to the requested public key"))
				return allErrors
			}
		}

		// All timestamps must be set.
		if newReq.Status.NotBefore == nil {
			allErrors = append(allErrors, field.Required(notBeforePath, "must be present and consistent with the issued certificate"))
		}
		if newReq.Status.NotAfter == nil {
			allErrors = append(allErrors, field.Required(notAfterPath, "must be present and consistent with the issued certificate"))
		}
		if newReq.Status.BeginRefreshAt == nil {
			allErrors = append(allErrors, field.Required(beginRefreshPath, "must be present and in the range [notbefore+10min, notafter-10min]"))
		}
		if len(allErrors) > 0 {
			return allErrors
		}

		// Validate that NotBefore is consistent with the status field, and within 5
		// minutes of the current time.
		if !newReq.Status.NotBefore.Time.Equal(leafCert.NotBefore) {
			allErrors = append(allErrors, field.Invalid(notBeforePath, newReq.Status.NotBefore.Time, "must be set to the NotBefore time encoded in the leaf certificate"))
			return allErrors
		}
		if !timeNear(newReq.Status.NotBefore.Time, clock.Now(), 5*time.Minute) {
			allErrors = append(allErrors, field.Invalid(notBeforePath, newReq.Status.NotBefore.Time, "must be set to within 5 minutes of kube-apiserver's current time"))
			return allErrors
		}

		// Validate that NotAfter is consistent with the status field
		if !newReq.Status.NotAfter.Time.Equal(leafCert.NotAfter) {
			allErrors = append(allErrors, field.Invalid(notAfterPath, newReq.Status.NotAfter.Time, "must be set to the NotAfter time encoded in the leaf certificate"))
			return allErrors
		}

		// Validate that leaf cert lifetime against minimum and maximum constraints.
		lifetime := leafCert.NotAfter.Sub(leafCert.NotBefore)
		if lifetime < 1*time.Hour {
			allErrors = append(allErrors, field.Invalid(certChainPath, lifetime, "leaf certificate lifetime must be >= 1 hour"))
			return allErrors
		}
		if lifetime > time.Duration(*newReq.Spec.MaxExpirationSeconds)*time.Second {
			allErrors = append(allErrors, field.Invalid(certChainPath, lifetime, fmt.Sprintf("leaf certificate lifetime must be <= spec.maxExpirationSeconds (%v)", *newReq.Spec.MaxExpirationSeconds)))
			return allErrors
		}

		// Validate that BeginRefreshAt is within limits.
		if newReq.Status.BeginRefreshAt.Time.Before(newReq.Status.NotBefore.Time.Add(10 * time.Minute)) {
			allErrors = append(allErrors, field.Invalid(beginRefreshPath, newReq.Status.BeginRefreshAt.Time, "must be at least 10 minutes after status.notBefore"))
			return allErrors
		}
		if newReq.Status.BeginRefreshAt.Time.After(newReq.Status.NotAfter.Time.Add(-10 * time.Minute)) {
			allErrors = append(allErrors, field.Invalid(beginRefreshPath, newReq.Status.BeginRefreshAt.Time, "must be at least 10 minutes before status.notAfter"))
			return allErrors
		}

		// Check the remainder of the certificates in the chain, if any.  We cannot
		// easily verify the chain, because the Golang X.509 libraries are wisely
		// written to prevent us from doing stupid things like verifying a partial
		// chain, but we can at least check that they are valid certificates.
		for {
			var nextBlock *pem.Block
			nextBlock, rest = pem.Decode(rest)
			if nextBlock == nil {
				break
			}

			if nextBlock.Type != "CERTIFICATE" {
				allErrors = append(allErrors, field.Invalid(certChainPath, newReq.Status.CertificateChain, "issued certificate chain must consist entirely of CERTFICATE PEM blocks"))
				return allErrors
			}

			_, err := x509.ParseCertificate(nextBlock.Bytes)
			if err != nil {
				allErrors = append(allErrors, field.Invalid(certChainPath, newReq.Status.CertificateChain, "intermediate certificate does not parse as valid X.509"))
				return allErrors
			}
		}

		return allErrors
	}

	// We are not transitioning to any terminal state.  The whole status object
	// is immutable.
	allErrors = append(allErrors, validateSemanticEquality(newReq.Status, oldReq.Status, field.NewPath("status"), `status is immutable unless transitioning to "Issued", "Denied", or "Failed"`)...)
	return allErrors
}

func pcrIsIssued(pcr *certificates.PodCertificateRequest) bool {
	for _, cond := range pcr.Status.Conditions {
		if cond.Type == certificates.PodCertificateRequestConditionTypeIssued && cond.Status == metav1.ConditionTrue {
			return true
		}
	}
	return false
}

func pcrIsDenied(pcr *certificates.PodCertificateRequest) bool {
	for _, cond := range pcr.Status.Conditions {
		if cond.Type == certificates.PodCertificateRequestConditionTypeDenied && cond.Status == metav1.ConditionTrue {
			return true
		}
	}
	return false
}

func pcrIsFailed(pcr *certificates.PodCertificateRequest) bool {
	for _, cond := range pcr.Status.Conditions {
		if cond.Type == certificates.PodCertificateRequestConditionTypeFailed && cond.Status == metav1.ConditionTrue {
			return true
		}
	}
	return false
}

func formatIndex(i int) string {
	return "[" + strconv.Itoa(i) + "]"
}

// Similar to apivalidation.ValidateImmutableField but we can supply our own detail string.
func validateSemanticEquality(oldVal, newVal any, fldPath *field.Path, detail string) field.ErrorList {
	allErrs := field.ErrorList{}
	if !apiequality.Semantic.DeepEqual(oldVal, newVal) {
		allErrs = append(allErrs, field.Invalid(fldPath, field.OmitValueType{}, detail))
	}
	return allErrs
}

func timeNear(a, b time.Time, skew time.Duration) bool {
	return a.After(b.Add(-skew)) && a.Before(b.Add(skew))
}
