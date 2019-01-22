// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package pkix contains shared, low level structures used for ASN.1 parsing
// and serialization of X.509 certificates, CRL and OCSP.
package pkix

import (
	// START CT CHANGES
	"encoding/hex"
	"fmt"

	"github.com/google/certificate-transparency-go/asn1"
	// END CT CHANGES
	"math/big"
	"time"
)

// AlgorithmIdentifier represents the ASN.1 structure of the same name. See RFC
// 5280, section 4.1.1.2.
type AlgorithmIdentifier struct {
	Algorithm  asn1.ObjectIdentifier
	Parameters asn1.RawValue `asn1:"optional"`
}

type RDNSequence []RelativeDistinguishedNameSET

var attributeTypeNames = map[string]string{
	"2.5.4.6":  "C",
	"2.5.4.10": "O",
	"2.5.4.11": "OU",
	"2.5.4.3":  "CN",
	"2.5.4.5":  "SERIALNUMBER",
	"2.5.4.7":  "L",
	"2.5.4.8":  "ST",
	"2.5.4.9":  "STREET",
	"2.5.4.17": "POSTALCODE",
}

// String returns a string representation of the sequence r,
// roughly following the RFC 2253 Distinguished Names syntax.
func (r RDNSequence) String() string {
	s := ""
	for i := 0; i < len(r); i++ {
		rdn := r[len(r)-1-i]
		if i > 0 {
			s += ","
		}
		for j, tv := range rdn {
			if j > 0 {
				s += "+"
			}

			oidString := tv.Type.String()
			typeName, ok := attributeTypeNames[oidString]
			if !ok {
				derBytes, err := asn1.Marshal(tv.Value)
				if err == nil {
					s += oidString + "=#" + hex.EncodeToString(derBytes)
					continue // No value escaping necessary.
				}

				typeName = oidString
			}

			valueString := fmt.Sprint(tv.Value)
			escaped := make([]rune, 0, len(valueString))

			for k, c := range valueString {
				escape := false

				switch c {
				case ',', '+', '"', '\\', '<', '>', ';':
					escape = true

				case ' ':
					escape = k == 0 || k == len(valueString)-1

				case '#':
					escape = k == 0
				}

				if escape {
					escaped = append(escaped, '\\', c)
				} else {
					escaped = append(escaped, c)
				}
			}

			s += typeName + "=" + string(escaped)
		}
	}

	return s
}

type RelativeDistinguishedNameSET []AttributeTypeAndValue

// AttributeTypeAndValue mirrors the ASN.1 structure of the same name in
// http://tools.ietf.org/html/rfc5280#section-4.1.2.4
type AttributeTypeAndValue struct {
	Type  asn1.ObjectIdentifier
	Value interface{}
}

// AttributeTypeAndValueSET represents a set of ASN.1 sequences of
// AttributeTypeAndValue sequences from RFC 2986 (PKCS #10).
type AttributeTypeAndValueSET struct {
	Type  asn1.ObjectIdentifier
	Value [][]AttributeTypeAndValue `asn1:"set"`
}

// Extension represents the ASN.1 structure of the same name. See RFC
// 5280, section 4.2.
type Extension struct {
	Id       asn1.ObjectIdentifier
	Critical bool `asn1:"optional"`
	Value    []byte
}

// Name represents an X.509 distinguished name. This only includes the common
// elements of a DN. When parsing, all elements are stored in Names and
// non-standard elements can be extracted from there. When marshaling, elements
// in ExtraNames are appended and override other values with the same OID.
type Name struct {
	Country, Organization, OrganizationalUnit []string
	Locality, Province                        []string
	StreetAddress, PostalCode                 []string
	SerialNumber, CommonName                  string

	Names      []AttributeTypeAndValue
	ExtraNames []AttributeTypeAndValue
}

func (n *Name) FillFromRDNSequence(rdns *RDNSequence) {
	for _, rdn := range *rdns {
		if len(rdn) == 0 {
			continue
		}

		for _, atv := range rdn {
			n.Names = append(n.Names, atv)
			value, ok := atv.Value.(string)
			if !ok {
				continue
			}

			t := atv.Type
			if len(t) == 4 && t[0] == OIDAttribute[0] && t[1] == OIDAttribute[1] && t[2] == OIDAttribute[2] {
				switch t[3] {
				case OIDCommonName[3]:
					n.CommonName = value
				case OIDSerialNumber[3]:
					n.SerialNumber = value
				case OIDCountry[3]:
					n.Country = append(n.Country, value)
				case OIDLocality[3]:
					n.Locality = append(n.Locality, value)
				case OIDProvince[3]:
					n.Province = append(n.Province, value)
				case OIDStreetAddress[3]:
					n.StreetAddress = append(n.StreetAddress, value)
				case OIDOrganization[3]:
					n.Organization = append(n.Organization, value)
				case OIDOrganizationalUnit[3]:
					n.OrganizationalUnit = append(n.OrganizationalUnit, value)
				case OIDPostalCode[3]:
					n.PostalCode = append(n.PostalCode, value)
				}
			}
		}
	}
}

var (
	OIDAttribute          = asn1.ObjectIdentifier{2, 5, 4}
	OIDCountry            = asn1.ObjectIdentifier{2, 5, 4, 6}
	OIDOrganization       = asn1.ObjectIdentifier{2, 5, 4, 10}
	OIDOrganizationalUnit = asn1.ObjectIdentifier{2, 5, 4, 11}
	OIDCommonName         = asn1.ObjectIdentifier{2, 5, 4, 3}
	OIDSerialNumber       = asn1.ObjectIdentifier{2, 5, 4, 5}
	OIDLocality           = asn1.ObjectIdentifier{2, 5, 4, 7}
	OIDProvince           = asn1.ObjectIdentifier{2, 5, 4, 8}
	OIDStreetAddress      = asn1.ObjectIdentifier{2, 5, 4, 9}
	OIDPostalCode         = asn1.ObjectIdentifier{2, 5, 4, 17}

	OIDPseudonym           = asn1.ObjectIdentifier{2, 5, 4, 65}
	OIDTitle               = asn1.ObjectIdentifier{2, 5, 4, 12}
	OIDDnQualifier         = asn1.ObjectIdentifier{2, 5, 4, 46}
	OIDName                = asn1.ObjectIdentifier{2, 5, 4, 41}
	OIDSurname             = asn1.ObjectIdentifier{2, 5, 4, 4}
	OIDGivenName           = asn1.ObjectIdentifier{2, 5, 4, 42}
	OIDInitials            = asn1.ObjectIdentifier{2, 5, 4, 43}
	OIDGenerationQualifier = asn1.ObjectIdentifier{2, 5, 4, 44}
)

// appendRDNs appends a relativeDistinguishedNameSET to the given RDNSequence
// and returns the new value. The relativeDistinguishedNameSET contains an
// attributeTypeAndValue for each of the given values. See RFC 5280, A.1, and
// search for AttributeTypeAndValue.
func (n Name) appendRDNs(in RDNSequence, values []string, oid asn1.ObjectIdentifier) RDNSequence {
	if len(values) == 0 || oidInAttributeTypeAndValue(oid, n.ExtraNames) {
		return in
	}

	s := make([]AttributeTypeAndValue, len(values))
	for i, value := range values {
		s[i].Type = oid
		s[i].Value = value
	}

	return append(in, s)
}

func (n Name) ToRDNSequence() (ret RDNSequence) {
	ret = n.appendRDNs(ret, n.Country, OIDCountry)
	ret = n.appendRDNs(ret, n.Province, OIDProvince)
	ret = n.appendRDNs(ret, n.Locality, OIDLocality)
	ret = n.appendRDNs(ret, n.StreetAddress, OIDStreetAddress)
	ret = n.appendRDNs(ret, n.PostalCode, OIDPostalCode)
	ret = n.appendRDNs(ret, n.Organization, OIDOrganization)
	ret = n.appendRDNs(ret, n.OrganizationalUnit, OIDOrganizationalUnit)
	if len(n.CommonName) > 0 {
		ret = n.appendRDNs(ret, []string{n.CommonName}, OIDCommonName)
	}
	if len(n.SerialNumber) > 0 {
		ret = n.appendRDNs(ret, []string{n.SerialNumber}, OIDSerialNumber)
	}
	for _, atv := range n.ExtraNames {
		ret = append(ret, []AttributeTypeAndValue{atv})
	}

	return ret
}

// String returns the string form of n, roughly following
// the RFC 2253 Distinguished Names syntax.
func (n Name) String() string {
	return n.ToRDNSequence().String()
}

// oidInAttributeTypeAndValue returns whether a type with the given OID exists
// in atv.
func oidInAttributeTypeAndValue(oid asn1.ObjectIdentifier, atv []AttributeTypeAndValue) bool {
	for _, a := range atv {
		if a.Type.Equal(oid) {
			return true
		}
	}
	return false
}

// CertificateList represents the ASN.1 structure of the same name. See RFC
// 5280, section 5.1. Use Certificate.CheckCRLSignature to verify the
// signature.
type CertificateList struct {
	TBSCertList        TBSCertificateList
	SignatureAlgorithm AlgorithmIdentifier
	SignatureValue     asn1.BitString
}

// HasExpired reports whether certList should have been updated by now.
func (certList *CertificateList) HasExpired(now time.Time) bool {
	return !now.Before(certList.TBSCertList.NextUpdate)
}

// TBSCertificateList represents the ASN.1 structure TBSCertList. See RFC
// 5280, section 5.1.
type TBSCertificateList struct {
	Raw                 asn1.RawContent
	Version             int `asn1:"optional,default:0"`
	Signature           AlgorithmIdentifier
	Issuer              RDNSequence
	ThisUpdate          time.Time
	NextUpdate          time.Time            `asn1:"optional"`
	RevokedCertificates []RevokedCertificate `asn1:"optional"`
	Extensions          []Extension          `asn1:"tag:0,optional,explicit"`
}

// RevokedCertificate represents the unnamed ASN.1 structure that makes up the
// revokedCertificates member of the TBSCertList structure. See RFC
// 5280, section 5.1.
type RevokedCertificate struct {
	SerialNumber   *big.Int
	RevocationTime time.Time
	Extensions     []Extension `asn1:"optional"`
}
