// Copyright 2017 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package x509

import (
	"bytes"
	"encoding/pem"
	"time"

	"github.com/google/certificate-transparency-go/asn1"
	"github.com/google/certificate-transparency-go/x509/pkix"
)

// OID values for CRL extensions (TBSCertList.Extensions), RFC 5280 s5.2.
var (
	OIDExtensionCRLNumber                = asn1.ObjectIdentifier{2, 5, 29, 20}
	OIDExtensionDeltaCRLIndicator        = asn1.ObjectIdentifier{2, 5, 29, 27}
	OIDExtensionIssuingDistributionPoint = asn1.ObjectIdentifier{2, 5, 29, 28}
)

// OID values for CRL entry extensions (RevokedCertificate.Extensions), RFC 5280 s5.3
var (
	OIDExtensionCRLReasons        = asn1.ObjectIdentifier{2, 5, 29, 21}
	OIDExtensionInvalidityDate    = asn1.ObjectIdentifier{2, 5, 29, 24}
	OIDExtensionCertificateIssuer = asn1.ObjectIdentifier{2, 5, 29, 29}
)

// RevocationReasonCode represents the reason for a certificate revocation; see RFC 5280 s5.3.1.
type RevocationReasonCode asn1.Enumerated

// RevocationReasonCode values.
var (
	Unspecified          = RevocationReasonCode(0)
	KeyCompromise        = RevocationReasonCode(1)
	CACompromise         = RevocationReasonCode(2)
	AffiliationChanged   = RevocationReasonCode(3)
	Superseded           = RevocationReasonCode(4)
	CessationOfOperation = RevocationReasonCode(5)
	CertificateHold      = RevocationReasonCode(6)
	RemoveFromCRL        = RevocationReasonCode(8)
	PrivilegeWithdrawn   = RevocationReasonCode(9)
	AACompromise         = RevocationReasonCode(10)
)

// ReasonFlag holds a bitmask of applicable revocation reasons, from RFC 5280 s4.2.1.13
type ReasonFlag int

// ReasonFlag values.
const (
	UnusedFlag ReasonFlag = 1 << iota
	KeyCompromiseFlag
	CACompromiseFlag
	AffiliationChangedFlag
	SupersededFlag
	CessationOfOperationFlag
	CertificateHoldFlag
	PrivilegeWithdrawnFlag
	AACompromiseFlag
)

// CertificateList represents the ASN.1 structure of the same name from RFC 5280, s5.1.
// It has the same content as pkix.CertificateList, but the contents include parsed versions
// of any extensions.
type CertificateList struct {
	Raw                asn1.RawContent
	TBSCertList        TBSCertList
	SignatureAlgorithm pkix.AlgorithmIdentifier
	SignatureValue     asn1.BitString
}

// ExpiredAt reports whether now is past the expiry time of certList.
func (certList *CertificateList) ExpiredAt(now time.Time) bool {
	return now.After(certList.TBSCertList.NextUpdate)
}

// Indication of whether extensions need to be critical or non-critical. Extensions that
// can be either are omitted from the map.
var listExtCritical = map[string]bool{
	// From RFC 5280...
	OIDExtensionAuthorityKeyId.String():           false, // s5.2.1
	OIDExtensionIssuerAltName.String():            false, // s5.2.2
	OIDExtensionCRLNumber.String():                false, // s5.2.3
	OIDExtensionDeltaCRLIndicator.String():        true,  // s5.2.4
	OIDExtensionIssuingDistributionPoint.String(): true,  // s5.2.5
	OIDExtensionFreshestCRL.String():              false, // s5.2.6
	OIDExtensionAuthorityInfoAccess.String():      false, // s5.2.7
}

var certExtCritical = map[string]bool{
	// From RFC 5280...
	OIDExtensionCRLReasons.String():        false, // s5.3.1
	OIDExtensionInvalidityDate.String():    false, // s5.3.2
	OIDExtensionCertificateIssuer.String(): true,  // s5.3.3
}

// IssuingDistributionPoint represents the ASN.1 structure of the same
// name
type IssuingDistributionPoint struct {
	DistributionPoint          distributionPointName `asn1:"optional,tag:0"`
	OnlyContainsUserCerts      bool                  `asn1:"optional,tag:1"`
	OnlyContainsCACerts        bool                  `asn1:"optional,tag:2"`
	OnlySomeReasons            asn1.BitString        `asn1:"optional,tag:3"`
	IndirectCRL                bool                  `asn1:"optional,tag:4"`
	OnlyContainsAttributeCerts bool                  `asn1:"optional,tag:5"`
}

// TBSCertList represents the ASN.1 structure of the same name from RFC
// 5280, section 5.1.  It has the same content as pkix.TBSCertificateList
// but the extensions are included in a parsed format.
type TBSCertList struct {
	Raw                 asn1.RawContent
	Version             int
	Signature           pkix.AlgorithmIdentifier
	Issuer              pkix.RDNSequence
	ThisUpdate          time.Time
	NextUpdate          time.Time
	RevokedCertificates []*RevokedCertificate
	Extensions          []pkix.Extension
	// Cracked out extensions:
	AuthorityKeyID               []byte
	IssuerAltNames               GeneralNames
	CRLNumber                    int
	BaseCRLNumber                int // -1 if no delta CRL present
	IssuingDistributionPoint     IssuingDistributionPoint
	IssuingDPFullNames           GeneralNames
	FreshestCRLDistributionPoint []string
	OCSPServer                   []string
	IssuingCertificateURL        []string
}

// ParseCertificateList parses a CertificateList (e.g. a CRL) from the given
// bytes. It's often the case that PEM encoded CRLs will appear where they
// should be DER encoded, so this function will transparently handle PEM
// encoding as long as there isn't any leading garbage.
func ParseCertificateList(clBytes []byte) (*CertificateList, error) {
	if bytes.HasPrefix(clBytes, pemCRLPrefix) {
		block, _ := pem.Decode(clBytes)
		if block != nil && block.Type == pemType {
			clBytes = block.Bytes
		}
	}
	return ParseCertificateListDER(clBytes)
}

// ParseCertificateListDER parses a DER encoded CertificateList from the given bytes.
// For non-fatal errors, this function returns both an error and a CertificateList
// object.
func ParseCertificateListDER(derBytes []byte) (*CertificateList, error) {
	var errs Errors
	// First parse the DER into the pkix structures.
	pkixList := new(pkix.CertificateList)
	if rest, err := asn1.Unmarshal(derBytes, pkixList); err != nil {
		errs.AddID(ErrInvalidCertList, err)
		return nil, &errs
	} else if len(rest) != 0 {
		errs.AddID(ErrTrailingCertList)
		return nil, &errs
	}

	// Transcribe the revoked certs but crack out extensions.
	revokedCerts := make([]*RevokedCertificate, len(pkixList.TBSCertList.RevokedCertificates))
	for i, pkixRevoked := range pkixList.TBSCertList.RevokedCertificates {
		revokedCerts[i] = parseRevokedCertificate(pkixRevoked, &errs)
		if revokedCerts[i] == nil {
			return nil, &errs
		}
	}

	certList := CertificateList{
		Raw: derBytes,
		TBSCertList: TBSCertList{
			Raw:                 pkixList.TBSCertList.Raw,
			Version:             pkixList.TBSCertList.Version,
			Signature:           pkixList.TBSCertList.Signature,
			Issuer:              pkixList.TBSCertList.Issuer,
			ThisUpdate:          pkixList.TBSCertList.ThisUpdate,
			NextUpdate:          pkixList.TBSCertList.NextUpdate,
			RevokedCertificates: revokedCerts,
			Extensions:          pkixList.TBSCertList.Extensions,
			CRLNumber:           -1,
			BaseCRLNumber:       -1,
		},
		SignatureAlgorithm: pkixList.SignatureAlgorithm,
		SignatureValue:     pkixList.SignatureValue,
	}

	// Now crack out extensions.
	for _, e := range certList.TBSCertList.Extensions {
		if expectCritical, present := listExtCritical[e.Id.String()]; present {
			if e.Critical && !expectCritical {
				errs.AddID(ErrUnexpectedlyCriticalCertListExtension, e.Id)
			} else if !e.Critical && expectCritical {
				errs.AddID(ErrUnexpectedlyNonCriticalCertListExtension, e.Id)
			}
		}
		switch {
		case e.Id.Equal(OIDExtensionAuthorityKeyId):
			// RFC 5280 s5.2.1
			var a authKeyId
			if rest, err := asn1.Unmarshal(e.Value, &a); err != nil {
				errs.AddID(ErrInvalidCertListAuthKeyID, err)
			} else if len(rest) != 0 {
				errs.AddID(ErrTrailingCertListAuthKeyID)
			}
			certList.TBSCertList.AuthorityKeyID = a.Id
		case e.Id.Equal(OIDExtensionIssuerAltName):
			// RFC 5280 s5.2.2
			if err := parseGeneralNames(e.Value, &certList.TBSCertList.IssuerAltNames); err != nil {
				errs.AddID(ErrInvalidCertListIssuerAltName, err)
			}
		case e.Id.Equal(OIDExtensionCRLNumber):
			// RFC 5280 s5.2.3
			if rest, err := asn1.Unmarshal(e.Value, &certList.TBSCertList.CRLNumber); err != nil {
				errs.AddID(ErrInvalidCertListCRLNumber, err)
			} else if len(rest) != 0 {
				errs.AddID(ErrTrailingCertListCRLNumber)
			}
			if certList.TBSCertList.CRLNumber < 0 {
				errs.AddID(ErrNegativeCertListCRLNumber, certList.TBSCertList.CRLNumber)
			}
		case e.Id.Equal(OIDExtensionDeltaCRLIndicator):
			// RFC 5280 s5.2.4
			if rest, err := asn1.Unmarshal(e.Value, &certList.TBSCertList.BaseCRLNumber); err != nil {
				errs.AddID(ErrInvalidCertListDeltaCRL, err)
			} else if len(rest) != 0 {
				errs.AddID(ErrTrailingCertListDeltaCRL)
			}
			if certList.TBSCertList.BaseCRLNumber < 0 {
				errs.AddID(ErrNegativeCertListDeltaCRL, certList.TBSCertList.BaseCRLNumber)
			}
		case e.Id.Equal(OIDExtensionIssuingDistributionPoint):
			parseIssuingDistributionPoint(e.Value, &certList.TBSCertList.IssuingDistributionPoint, &certList.TBSCertList.IssuingDPFullNames, &errs)
		case e.Id.Equal(OIDExtensionFreshestCRL):
			// RFC 5280 s5.2.6
			if err := parseDistributionPoints(e.Value, &certList.TBSCertList.FreshestCRLDistributionPoint); err != nil {
				errs.AddID(ErrInvalidCertListFreshestCRL, err)
				return nil, err
			}
		case e.Id.Equal(OIDExtensionAuthorityInfoAccess):
			// RFC 5280 s5.2.7
			var aia []accessDescription
			if rest, err := asn1.Unmarshal(e.Value, &aia); err != nil {
				errs.AddID(ErrInvalidCertListAuthInfoAccess, err)
			} else if len(rest) != 0 {
				errs.AddID(ErrTrailingCertListAuthInfoAccess)
			}

			for _, v := range aia {
				// GeneralName: uniformResourceIdentifier [6] IA5String
				if v.Location.Tag != tagURI {
					continue
				}
				switch {
				case v.Method.Equal(OIDAuthorityInfoAccessOCSP):
					certList.TBSCertList.OCSPServer = append(certList.TBSCertList.OCSPServer, string(v.Location.Bytes))
				case v.Method.Equal(OIDAuthorityInfoAccessIssuers):
					certList.TBSCertList.IssuingCertificateURL = append(certList.TBSCertList.IssuingCertificateURL, string(v.Location.Bytes))
				}
				// TODO(drysdale): cope with more possibilities
			}
		default:
			if e.Critical {
				errs.AddID(ErrUnhandledCriticalCertListExtension, e.Id)
			}
		}
	}

	if errs.Fatal() {
		return nil, &errs
	}
	if errs.Empty() {
		return &certList, nil
	}
	return &certList, &errs
}

func parseIssuingDistributionPoint(data []byte, idp *IssuingDistributionPoint, name *GeneralNames, errs *Errors) {
	// RFC 5280 s5.2.5
	if rest, err := asn1.Unmarshal(data, idp); err != nil {
		errs.AddID(ErrInvalidCertListIssuingDP, err)
	} else if len(rest) != 0 {
		errs.AddID(ErrTrailingCertListIssuingDP)
	}

	typeCount := 0
	if idp.OnlyContainsUserCerts {
		typeCount++
	}
	if idp.OnlyContainsCACerts {
		typeCount++
	}
	if idp.OnlyContainsAttributeCerts {
		typeCount++
	}
	if typeCount > 1 {
		errs.AddID(ErrCertListIssuingDPMultipleTypes, idp.OnlyContainsUserCerts, idp.OnlyContainsCACerts, idp.OnlyContainsAttributeCerts)
	}
	for _, fn := range idp.DistributionPoint.FullName {
		if _, err := parseGeneralName(fn.FullBytes, name, false); err != nil {
			errs.AddID(ErrCertListIssuingDPInvalidFullName, err)
		}
	}
}

// RevokedCertificate represents the unnamed ASN.1 structure that makes up the
// revokedCertificates member of the TBSCertList structure from RFC 5280, s5.1.
// It has the same content as pkix.RevokedCertificate but the extensions are
// included in a parsed format.
type RevokedCertificate struct {
	pkix.RevokedCertificate
	// Cracked out extensions:
	RevocationReason RevocationReasonCode
	InvalidityDate   time.Time
	Issuer           GeneralNames
}

func parseRevokedCertificate(pkixRevoked pkix.RevokedCertificate, errs *Errors) *RevokedCertificate {
	result := RevokedCertificate{RevokedCertificate: pkixRevoked}
	for _, e := range pkixRevoked.Extensions {
		if expectCritical, present := certExtCritical[e.Id.String()]; present {
			if e.Critical && !expectCritical {
				errs.AddID(ErrUnexpectedlyCriticalRevokedCertExtension, e.Id)
			} else if !e.Critical && expectCritical {
				errs.AddID(ErrUnexpectedlyNonCriticalRevokedCertExtension, e.Id)
			}
		}
		switch {
		case e.Id.Equal(OIDExtensionCRLReasons):
			// RFC 5280, s5.3.1
			var reason asn1.Enumerated
			if rest, err := asn1.Unmarshal(e.Value, &reason); err != nil {
				errs.AddID(ErrInvalidRevocationReason, err)
			} else if len(rest) != 0 {
				errs.AddID(ErrTrailingRevocationReason)
			}
			result.RevocationReason = RevocationReasonCode(reason)
		case e.Id.Equal(OIDExtensionInvalidityDate):
			// RFC 5280, s5.3.2
			if rest, err := asn1.Unmarshal(e.Value, &result.InvalidityDate); err != nil {
				errs.AddID(ErrInvalidRevocationInvalidityDate, err)
			} else if len(rest) != 0 {
				errs.AddID(ErrTrailingRevocationInvalidityDate)
			}
		case e.Id.Equal(OIDExtensionCertificateIssuer):
			// RFC 5280, s5.3.3
			if err := parseGeneralNames(e.Value, &result.Issuer); err != nil {
				errs.AddID(ErrInvalidRevocationIssuer, err)
			}
		default:
			if e.Critical {
				errs.AddID(ErrUnhandledCriticalRevokedCertExtension, e.Id)
			}
		}
	}
	return &result
}

// CheckCertificateListSignature checks that the signature in crl is from c.
func (c *Certificate) CheckCertificateListSignature(crl *CertificateList) error {
	algo := SignatureAlgorithmFromAI(crl.SignatureAlgorithm)
	return c.CheckSignature(algo, crl.TBSCertList.Raw, crl.SignatureValue.RightAlign())
}
