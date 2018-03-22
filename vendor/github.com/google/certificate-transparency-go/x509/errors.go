package x509

import "fmt"

// To preserve error IDs, only append to this list, never insert.
const (
	ErrInvalidID ErrorID = iota
	ErrInvalidCertList
	ErrTrailingCertList
	ErrUnexpectedlyCriticalCertListExtension
	ErrUnexpectedlyNonCriticalCertListExtension
	ErrInvalidCertListAuthKeyID
	ErrTrailingCertListAuthKeyID
	ErrInvalidCertListIssuerAltName
	ErrInvalidCertListCRLNumber
	ErrTrailingCertListCRLNumber
	ErrNegativeCertListCRLNumber
	ErrInvalidCertListDeltaCRL
	ErrTrailingCertListDeltaCRL
	ErrNegativeCertListDeltaCRL
	ErrInvalidCertListIssuingDP
	ErrTrailingCertListIssuingDP
	ErrCertListIssuingDPMultipleTypes
	ErrCertListIssuingDPInvalidFullName
	ErrInvalidCertListFreshestCRL
	ErrInvalidCertListAuthInfoAccess
	ErrTrailingCertListAuthInfoAccess
	ErrUnhandledCriticalCertListExtension
	ErrUnexpectedlyCriticalRevokedCertExtension
	ErrUnexpectedlyNonCriticalRevokedCertExtension
	ErrInvalidRevocationReason
	ErrTrailingRevocationReason
	ErrInvalidRevocationInvalidityDate
	ErrTrailingRevocationInvalidityDate
	ErrInvalidRevocationIssuer
	ErrUnhandledCriticalRevokedCertExtension

	ErrMaxID
)

// idToError gives a template x509.Error for each defined ErrorID; where the Summary
// field may hold format specifiers that take field parameters.
var idToError map[ErrorID]Error

var errorInfo = []Error{
	{
		ID:       ErrInvalidCertList,
		Summary:  "x509: failed to parse CertificateList: %v",
		Field:    "CertificateList",
		SpecRef:  "RFC 5280 s5.1",
		Category: InvalidASN1Content,
		Fatal:    true,
	},
	{
		ID:       ErrTrailingCertList,
		Summary:  "x509: trailing data after CertificateList",
		Field:    "CertificateList",
		SpecRef:  "RFC 5280 s5.1",
		Category: InvalidASN1Content,
		Fatal:    true,
	},

	{
		ID:       ErrUnexpectedlyCriticalCertListExtension,
		Summary:  "x509: certificate list extension %v marked critical but expected to be non-critical",
		Field:    "tbsCertList.crlExtensions.*.critical",
		SpecRef:  "RFC 5280 s5.2",
		Category: MalformedCRL,
	},
	{
		ID:       ErrUnexpectedlyNonCriticalCertListExtension,
		Summary:  "x509: certificate list extension %v marked non-critical but expected to be critical",
		Field:    "tbsCertList.crlExtensions.*.critical",
		SpecRef:  "RFC 5280 s5.2",
		Category: MalformedCRL,
	},

	{
		ID:       ErrInvalidCertListAuthKeyID,
		Summary:  "x509: failed to unmarshal certificate-list authority key-id: %v",
		Field:    "tbsCertList.crlExtensions.*.AuthorityKeyIdentifier",
		SpecRef:  "RFC 5280 s5.2.1",
		Category: InvalidASN1Content,
		Fatal:    true,
	},
	{
		ID:       ErrTrailingCertListAuthKeyID,
		Summary:  "x509: trailing data after certificate list auth key ID",
		Field:    "tbsCertList.crlExtensions.*.AuthorityKeyIdentifier",
		SpecRef:  "RFC 5280 s5.2.1",
		Category: InvalidASN1Content,
		Fatal:    true,
	},
	{
		ID:       ErrInvalidCertListIssuerAltName,
		Summary:  "x509: failed to parse CRL issuer alt name: %v",
		Field:    "tbsCertList.crlExtensions.*.IssuerAltName",
		SpecRef:  "RFC 5280 s5.2.2",
		Category: InvalidASN1Content,
		Fatal:    true,
	},
	{
		ID:       ErrInvalidCertListCRLNumber,
		Summary:  "x509: failed to unmarshal certificate-list crl-number: %v",
		Field:    "tbsCertList.crlExtensions.*.CRLNumber",
		SpecRef:  "RFC 5280 s5.2.3",
		Category: InvalidASN1Content,
		Fatal:    true,
	},
	{
		ID:       ErrTrailingCertListCRLNumber,
		Summary:  "x509: trailing data after certificate list crl-number",
		Field:    "tbsCertList.crlExtensions.*.CRLNumber",
		SpecRef:  "RFC 5280 s5.2.3",
		Category: InvalidASN1Content,
		Fatal:    true,
	},
	{
		ID:       ErrNegativeCertListCRLNumber,
		Summary:  "x509: negative certificate list crl-number: %d",
		Field:    "tbsCertList.crlExtensions.*.CRLNumber",
		SpecRef:  "RFC 5280 s5.2.3",
		Category: MalformedCRL,
		Fatal:    true,
	},
	{
		ID:       ErrInvalidCertListDeltaCRL,
		Summary:  "x509: failed to unmarshal certificate-list delta-crl: %v",
		Field:    "tbsCertList.crlExtensions.*.BaseCRLNumber",
		SpecRef:  "RFC 5280 s5.2.4",
		Category: InvalidASN1Content,
		Fatal:    true,
	},
	{
		ID:       ErrTrailingCertListDeltaCRL,
		Summary:  "x509: trailing data after certificate list delta-crl",
		Field:    "tbsCertList.crlExtensions.*.BaseCRLNumber",
		SpecRef:  "RFC 5280 s5.2.4",
		Category: InvalidASN1Content,
		Fatal:    true,
	},
	{
		ID:       ErrNegativeCertListDeltaCRL,
		Summary:  "x509: negative certificate list base-crl-number: %d",
		Field:    "tbsCertList.crlExtensions.*.BaseCRLNumber",
		SpecRef:  "RFC 5280 s5.2.4",
		Category: MalformedCRL,
		Fatal:    true,
	},
	{
		ID:       ErrInvalidCertListIssuingDP,
		Summary:  "x509: failed to unmarshal certificate list issuing distribution point: %v",
		Field:    "tbsCertList.crlExtensions.*.IssuingDistributionPoint",
		SpecRef:  "RFC 5280 s5.2.5",
		Category: InvalidASN1Content,
		Fatal:    true,
	},
	{
		ID:       ErrTrailingCertListIssuingDP,
		Summary:  "x509: trailing data after certificate list issuing distribution point",
		Field:    "tbsCertList.crlExtensions.*.IssuingDistributionPoint",
		SpecRef:  "RFC 5280 s5.2.5",
		Category: InvalidASN1Content,
		Fatal:    true,
	},
	{
		ID:       ErrCertListIssuingDPMultipleTypes,
		Summary:  "x509: multiple cert types set in issuing-distribution-point: user:%v CA:%v attr:%v",
		Field:    "tbsCertList.crlExtensions.*.IssuingDistributionPoint",
		SpecRef:  "RFC 5280 s5.2.5",
		SpecText: "at most one of onlyContainsUserCerts, onlyContainsCACerts, and onlyContainsAttributeCerts may be set to TRUE.",
		Category: MalformedCRL,
		Fatal:    true,
	},
	{
		ID:       ErrCertListIssuingDPInvalidFullName,
		Summary:  "x509: failed to parse CRL issuing-distribution-point fullName: %v",
		Field:    "tbsCertList.crlExtensions.*.IssuingDistributionPoint.distributionPoint",
		SpecRef:  "RFC 5280 s5.2.5",
		Category: InvalidASN1Content,
		Fatal:    true,
	},
	{
		ID:       ErrInvalidCertListFreshestCRL,
		Summary:  "x509: failed to unmarshal certificate list freshestCRL: %v",
		Field:    "tbsCertList.crlExtensions.*.FreshestCRL",
		SpecRef:  "RFC 5280 s5.2.6",
		Category: InvalidASN1Content,
		Fatal:    true,
	},
	{
		ID:       ErrInvalidCertListAuthInfoAccess,
		Summary:  "x509: failed to unmarshal certificate list authority info access: %v",
		Field:    "tbsCertList.crlExtensions.*.AuthorityInfoAccess",
		SpecRef:  "RFC 5280 s5.2.7",
		Category: InvalidASN1Content,
		Fatal:    true,
	},
	{
		ID:       ErrTrailingCertListAuthInfoAccess,
		Summary:  "x509: trailing data after certificate list authority info access",
		Field:    "tbsCertList.crlExtensions.*.AuthorityInfoAccess",
		SpecRef:  "RFC 5280 s5.2.7",
		Category: InvalidASN1Content,
		Fatal:    true,
	},
	{
		ID:       ErrUnhandledCriticalCertListExtension,
		Summary:  "x509: unhandled critical extension in certificate list: %v",
		Field:    "tbsCertList.revokedCertificates.crlExtensions.*",
		SpecRef:  "RFC 5280 s5.2",
		SpecText: "If a CRL contains a critical extension that the application cannot process, then the application MUST NOT use that CRL to determine the status of certificates.",
		Category: MalformedCRL,
		Fatal:    true,
	},

	{
		ID:       ErrUnexpectedlyCriticalRevokedCertExtension,
		Summary:  "x509: revoked certificate extension %v marked critical but expected to be non-critical",
		Field:    "tbsCertList.revokedCertificates.crlEntryExtensions.*.critical",
		SpecRef:  "RFC 5280 s5.3",
		Category: MalformedCRL,
	},
	{
		ID:       ErrUnexpectedlyNonCriticalRevokedCertExtension,
		Summary:  "x509: revoked certificate extension %v marked non-critical but expected to be critical",
		Field:    "tbsCertList.revokedCertificates.crlEntryExtensions.*.critical",
		SpecRef:  "RFC 5280 s5.3",
		Category: MalformedCRL,
	},

	{
		ID:       ErrInvalidRevocationReason,
		Summary:  "x509: failed to parse revocation reason: %v",
		Field:    "tbsCertList.revokedCertificates.crlEntryExtensions.*.CRLReason",
		SpecRef:  "RFC 5280 s5.3.1",
		Category: InvalidASN1Content,
		Fatal:    true,
	},
	{
		ID:       ErrTrailingRevocationReason,
		Summary:  "x509: trailing data after revoked certificate reason",
		Field:    "tbsCertList.revokedCertificates.crlEntryExtensions.*.CRLReason",
		SpecRef:  "RFC 5280 s5.3.1",
		Category: InvalidASN1Content,
		Fatal:    true,
	},
	{
		ID:       ErrInvalidRevocationInvalidityDate,
		Summary:  "x509: failed to parse revoked certificate invalidity date: %v",
		Field:    "tbsCertList.revokedCertificates.crlEntryExtensions.*.InvalidityDate",
		SpecRef:  "RFC 5280 s5.3.2",
		Category: InvalidASN1Content,
		Fatal:    true,
	},
	{
		ID:       ErrTrailingRevocationInvalidityDate,
		Summary:  "x509: trailing data after revoked certificate invalidity date",
		Field:    "tbsCertList.revokedCertificates.crlEntryExtensions.*.InvalidityDate",
		SpecRef:  "RFC 5280 s5.3.2",
		Category: InvalidASN1Content,
		Fatal:    true,
	},
	{
		ID:       ErrInvalidRevocationIssuer,
		Summary:  "x509: failed to parse revocation issuer %v",
		Field:    "tbsCertList.revokedCertificates.crlEntryExtensions.*.CertificateIssuer",
		SpecRef:  "RFC 5280 s5.3.3",
		Category: InvalidASN1Content,
		Fatal:    true,
	},
	{
		ID:       ErrUnhandledCriticalRevokedCertExtension,
		Summary:  "x509: unhandled critical extension in revoked certificate: %v",
		Field:    "tbsCertList.revokedCertificates.crlEntryExtensions.*",
		SpecRef:  "RFC 5280 s5.3",
		SpecText: "If a CRL contains a critical CRL entry extension that the application cannot process, then the application MUST NOT use that CRL to determine the status of any certificates.",
		Category: MalformedCRL,
		Fatal:    true,
	},
}

func init() {
	idToError = make(map[ErrorID]Error, len(errorInfo))
	for _, info := range errorInfo {
		idToError[info.ID] = info
	}
}

// NewError builds a new x509.Error based on the template for the given id.
func NewError(id ErrorID, args ...interface{}) Error {
	var err Error
	if id >= ErrMaxID {
		err.ID = id
		err.Summary = fmt.Sprintf("Unknown error ID %v: args %+v", id, args)
		err.Fatal = true
	} else {
		err = idToError[id]
		err.Summary = fmt.Sprintf(err.Summary, args...)
	}
	return err
}
