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
	"time"

	"github.com/google/go-cmp/cmp" //nolint:depguard
	v1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
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
				conditionDiff := cmp.Diff(oldConditions, newConditions)
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

type ValidatePodCertificateRequestOptions struct {
	AllowSettingStatusFields bool
	SkipCryptoValidation     bool
	Clock                    clock.PassiveClock
}

// ValidatePodCertificateRequest runs all validation checks on a pod certificate request.
func ValidatePodCertificateRequest(req *certificates.PodCertificateRequest, opts ValidatePodCertificateRequestOptions) field.ErrorList {
	var allErrors field.ErrorList

	metaErrors := apivalidation.ValidateObjectMeta(&req.ObjectMeta, true, validatePodCertificateRequestName, field.NewPath("metadata"))
	allErrors = append(allErrors, metaErrors...)

	signerNameErrors := apivalidation.ValidateSignerName(field.NewPath("spec", "signerName"), req.Spec.SignerName)
	allErrors = append(allErrors, signerNameErrors...)

	// TODO(KEP-4317): Basic format validation on other spec fields.

	// TODO(KEP-4317): Denied / Failed conditions are not allowed if the
	// certificate is issued.

	// Everything below this point is crypto verification that should be skipped
	// if the fields haven't changed.
	if opts.SkipCryptoValidation {
		return allErrors
	}

	// TODO(KEP-4317): Check length of PKIXPublicKey and ProofOfPossession
	// fields.

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
			allErrors = append(allErrors, field.Invalid(popPath, req.Spec.ProofOfPossession, "could not verify proof-of-possession signature"))
			return allErrors
		}

	case *ecdsa.PublicKey:
		if pub.Curve != elliptic.P256() && pub.Curve != elliptic.P384() {
			allErrors = append(allErrors, field.Invalid(pkixPath, req.Spec.PKIXPublicKey, "elliptic public keys must use curve P256 or P384"))
			return allErrors
		}
		if !ecdsa.VerifyASN1(pub, hashBytes([]byte(req.Spec.PodUID)), req.Spec.ProofOfPossession) {
			allErrors = append(allErrors, field.Invalid(popPath, req.Spec.ProofOfPossession, "could not verify proof-of-possession signature"))
			return allErrors
		}

	case *rsa.PublicKey:
		if pub.Size() != 3072 && pub.Size() != 4096 {
			allErrors = append(allErrors, field.Invalid(pkixPath, req.Spec.PKIXPublicKey, "RSA keys must have modulus size 3072 or 4096"))
			return allErrors
		}
		if err := rsa.VerifyPKCS1v15(pub, crypto.SHA256, hashBytes([]byte(req.Spec.PodUID)), req.Spec.ProofOfPossession); err != nil {
			allErrors = append(allErrors, field.Invalid(popPath, req.Spec.ProofOfPossession, "could not verify proof-of-possession signature"))
			return allErrors
		}

	default:
		allErrors = append(allErrors, field.Invalid(pkixPath, req.Spec.PKIXPublicKey, "unknown public key type; supported types are Ed25519, ECDSA, and RSA"))
		return allErrors
	}

	// Bail out if the certificate chain hasn't been set.
	if req.Status.CertificateChain == "" {
		return allErrors
	}

	// TODO(KEP-4317): Check length of CertificateChain field.

	leafBlock, rest := pem.Decode([]byte(req.Status.CertificateChain))
	if leafBlock == nil {
		allErrors = append(allErrors, field.Invalid(certChainPath, req.Status.CertificateChain, "issued certificate chain must contain at least one certificate"))
		return allErrors
	}
	if leafBlock.Type != "CERTIFICATE" {
		allErrors = append(allErrors, field.Invalid(certChainPath, req.Status.CertificateChain, "issued certificate chain must consist entirely of CERTIFICATE PEM blocks"))
		return allErrors
	}

	leafCert, err := x509.ParseCertificate(leafBlock.Bytes)
	if err != nil {
		allErrors = append(allErrors, field.Invalid(certChainPath, req.Status.CertificateChain, "leaf certificate does not parse as valid X.509"))
		return allErrors
	}

	// Status.IssuedAt must be set, and within 5 minutes of the current time.
	if req.Status.IssuedAt == nil {
		allErrors = append(allErrors, field.Required(issuedAtPath, "must be present and set to a plausible time"))
		return allErrors
	}
	if !timeNear(req.Status.IssuedAt.Time, opts.Clock.Now(), 5*time.Minute) {
		allErrors = append(allErrors, field.Invalid(issuedAtPath, req.Status.IssuedAt.Time, "must be set to within 5 minutes of kube-apiserver's current time"))
		return allErrors
	}

	// Validate that NotBefore is consistent with the status field, and within 5
	// minutes of the current time.
	if req.Status.NotBefore == nil {
		allErrors = append(allErrors, field.Required(notBeforePath, "must be present and consistent with the issued certificate"))
		return allErrors
	}
	if !req.Status.NotBefore.Time.Equal(leafCert.NotBefore) {
		allErrors = append(allErrors, field.Invalid(notBeforePath, req.Status.NotBefore.Time, "must be set to the NotBefore time encoded in the leaf certificate"))
		return allErrors
	}
	if !timeNear(req.Status.NotBefore.Time, opts.Clock.Now(), 5*time.Minute) {
		allErrors = append(allErrors, field.Invalid(notBeforePath, req.Status.NotBefore.Time, "must be set to within 5 minutes of kube-apiserver's current time"))
		return allErrors
	}

	// Validate that NotAfter is consistent with the status field
	if req.Status.NotAfter == nil {
		allErrors = append(allErrors, field.Required(notAfterPath, "must be present and consistent with the the issued certificate"))
		return allErrors
	}
	if !req.Status.NotAfter.Time.Equal(leafCert.NotAfter) {
		allErrors = append(allErrors, field.Invalid(notAfterPath, req.Status.NotAfter.Time, "must be set to the NotAfter time encoded in the leaf certificate"))
		return allErrors
	}

	// Validate that leaf cert lifetime against minimum and maximum constraints.
	lifetime := leafCert.NotAfter.Sub(leafCert.NotBefore)
	if lifetime < 1*time.Hour {
		allErrors = append(allErrors, field.Invalid(certChainPath, lifetime, "leaf certificate lifetime must be >= 1 hour"))
		return allErrors
	}
	if req.Spec.MaxExpirationSeconds != nil && lifetime > time.Duration(*req.Spec.MaxExpirationSeconds)*time.Second {
		allErrors = append(allErrors, field.Invalid(certChainPath, lifetime, fmt.Sprintf("leaf certificate lifetime must be <= spec.maxExpirationSeconds (%v)", *req.Spec.MaxExpirationSeconds)))
		return allErrors
	}

	// Validate that BeginRefreshAt is within limits.
	if req.Status.BeginRefreshAt == nil {
		allErrors = append(allErrors, field.Required(beginRefreshPath, "must be present and at least 10 minutes before the certificate expires"))
		return allErrors
	}
	if req.Status.BeginRefreshAt.Time.Before(req.Status.NotBefore.Time.Add(10 * time.Minute)) {
		allErrors = append(allErrors, field.Invalid(beginRefreshPath, req.Status.BeginRefreshAt.Time, "must be at least 10 minutes after status.NotBefore"))
		return allErrors
	}
	if req.Status.BeginRefreshAt.Time.After(req.Status.NotAfter.Time.Add(-10 * time.Minute)) {
		allErrors = append(allErrors, field.Invalid(beginRefreshPath, req.Status.BeginRefreshAt.Time, "must be at least 10 minutes before status.notAfter"))
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
			allErrors = append(allErrors, field.Invalid(certChainPath, req.Status.CertificateChain, "issued certificate chain must consist entirely of CERTFICATE PEM blocks"))
			return allErrors
		}

		_, err := x509.ParseCertificate(nextBlock.Bytes)
		if err != nil {
			allErrors = append(allErrors, field.Invalid(certChainPath, req.Status.CertificateChain, "intermediate certificate does not parse as valid X.509"))
			return allErrors
		}
	}

	// TODO(KEP-4317): Tests

	return allErrors
}

func timeNear(a, b time.Time, skew time.Duration) bool {
	return a.After(b.Add(-skew)) && a.Before(b.Add(skew))
}

func hashBytes(in []byte) []byte {
	out := sha256.Sum256(in)
	return out[:]
}

var (
	pkixPath         = field.NewPath("spec", "pkixPublicKey")
	popPath          = field.NewPath("spec", "proofOfPossession")
	certChainPath    = field.NewPath("status", "certificateChain")
	issuedAtPath     = field.NewPath("status", "issuedAt")
	notBeforePath    = field.NewPath("status", "notBefore")
	notAfterPath     = field.NewPath("status", "notAfter")
	beginRefreshPath = field.NewPath("status", "beginRefreshAt")
)

// ValidatePodCertificateRequestUpdate runs all update validation checks on an
// update.
func ValidatePodCertificateRequestUpdate(newReq, oldReq *certificates.PodCertificateRequest, optsIn ValidatePodCertificateRequestOptions) field.ErrorList {
	opts := optsIn
	if bytes.Equal(newReq.Spec.PKIXPublicKey, oldReq.Spec.PKIXPublicKey) && bytes.Equal(newReq.Spec.ProofOfPossession, oldReq.Spec.ProofOfPossession) && newReq.Status.CertificateChain == oldReq.Status.CertificateChain {
		opts.SkipCryptoValidation = true
	}

	var allErrors field.ErrorList
	allErrors = append(allErrors, ValidatePodCertificateRequest(newReq, opts)...)
	allErrors = append(allErrors, apivalidation.ValidateObjectMetaUpdate(&newReq.ObjectMeta, &oldReq.ObjectMeta, field.NewPath("metadata"))...)

	// All spec fields are immutable.
	allErrors = append(allErrors, apivalidation.ValidateImmutableField(newReq.Spec.SignerName, oldReq.Spec.SignerName, field.NewPath("spec", "signerName"))...)
	allErrors = append(allErrors, apivalidation.ValidateImmutableField(newReq.Spec.PodName, oldReq.Spec.PodName, field.NewPath("spec", "podName"))...)
	allErrors = append(allErrors, apivalidation.ValidateImmutableField(newReq.Spec.PodUID, oldReq.Spec.PodUID, field.NewPath("spec", "podUID"))...)
	allErrors = append(allErrors, apivalidation.ValidateImmutableField(newReq.Spec.ServiceAccountName, oldReq.Spec.ServiceAccountName, field.NewPath("spec", "serviceAccountName"))...)
	allErrors = append(allErrors, apivalidation.ValidateImmutableField(newReq.Spec.ServiceAccountUID, oldReq.Spec.ServiceAccountUID, field.NewPath("spec", "serviceAccountUID"))...)
	allErrors = append(allErrors, apivalidation.ValidateImmutableField(newReq.Spec.NodeName, oldReq.Spec.NodeName, field.NewPath("spec", "nodeName"))...)
	allErrors = append(allErrors, apivalidation.ValidateImmutableField(newReq.Spec.NodeUID, oldReq.Spec.NodeUID, field.NewPath("spec", "nodeUID"))...)
	allErrors = append(allErrors, apivalidation.ValidateImmutableField(newReq.Spec.MaxExpirationSeconds, oldReq.Spec.MaxExpirationSeconds, field.NewPath("spec", "maxExpirationSeconds"))...)
	allErrors = append(allErrors, apivalidation.ValidateImmutableField(newReq.Spec.PKIXPublicKey, oldReq.Spec.PKIXPublicKey, field.NewPath("spec", "pkixPublicKey"))...)
	allErrors = append(allErrors, apivalidation.ValidateImmutableField(newReq.Spec.ProofOfPossession, oldReq.Spec.ProofOfPossession, field.NewPath("spec", "proofOfPossession"))...)
	// Backstop to catch any new fields we forgot to handle in validation.
	if len(allErrors) == 0 {
		allErrors = append(allErrors, apivalidation.ValidateImmutableField(newReq.Spec, oldReq.Spec, field.NewPath("spec"))...)
	}

	if !opts.AllowSettingStatusFields {
		allErrors = append(allErrors, apivalidation.ValidateImmutableField(newReq.Status.CertificateChain, oldReq.Status.CertificateChain, field.NewPath("status", "certificateChain"))...)
		allErrors = append(allErrors, apivalidation.ValidateImmutableField(newReq.Status.IssuedAt, oldReq.Status.IssuedAt, field.NewPath("status", "issuedAt"))...)
		allErrors = append(allErrors, apivalidation.ValidateImmutableField(newReq.Status.NotBefore, oldReq.Status.NotBefore, field.NewPath("status", "notBefore"))...)
		allErrors = append(allErrors, apivalidation.ValidateImmutableField(newReq.Status.BeginRefreshAt, oldReq.Status.BeginRefreshAt, field.NewPath("status", "beginRefreshAt"))...)
		allErrors = append(allErrors, apivalidation.ValidateImmutableField(newReq.Status.NotAfter, oldReq.Status.NotAfter, field.NewPath("status", "notAfter"))...)
	}

	return allErrors
}

// We don't care what you call your pod certificate requests.
func validatePodCertificateRequestName(name string, prefix bool) []string {
	// TODO(KEP-4317): Should we restrict PCR names beyond normal Kubernetes names?
	return nil
}

func ValidatePodCertificateRequestStatusUpdate(newReq, oldReq *certificates.PodCertificateRequest, opts ValidatePodCertificateRequestOptions) field.ErrorList {
	optsCopy := opts
	optsCopy.AllowSettingStatusFields = true

	return ValidatePodCertificateRequestUpdate(newReq, oldReq, optsCopy)
}
