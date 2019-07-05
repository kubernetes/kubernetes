// Package pkcs7 implements the subset of the CMS PKCS #7 datatype that is typically
// used to package certificates and CRLs.  Using openssl, every certificate converted
// to PKCS #7 format from another encoding such as PEM conforms to this implementation.
// reference: https://www.openssl.org/docs/man1.1.0/apps/crl2pkcs7.html
//
//			PKCS #7 Data type, reference: https://tools.ietf.org/html/rfc2315
//
// The full pkcs#7 cryptographic message syntax allows for cryptographic enhancements,
// for example data can be encrypted and signed and then packaged through pkcs#7 to be
// sent over a network and then verified and decrypted.  It is asn1, and the type of
// PKCS #7 ContentInfo, which comprises the PKCS #7 structure, is:
//
//			ContentInfo ::= SEQUENCE {
//				contentType ContentType,
//				content [0] EXPLICIT ANY DEFINED BY contentType OPTIONAL
//			}
//
// There are 6 possible ContentTypes, data, signedData, envelopedData,
// signedAndEnvelopedData, digestedData, and encryptedData.  Here signedData, Data, and encrypted
// Data are implemented, as the degenerate case of signedData without a signature is the typical
// format for transferring certificates and CRLS, and Data and encryptedData are used in PKCS #12
// formats.
// The ContentType signedData has the form:
//
//
//			signedData ::= SEQUENCE {
//				version Version,
//				digestAlgorithms DigestAlgorithmIdentifiers,
//				contentInfo ContentInfo,
//				certificates [0] IMPLICIT ExtendedCertificatesAndCertificates OPTIONAL
//				crls [1] IMPLICIT CertificateRevocationLists OPTIONAL,
//				signerInfos SignerInfos
//			}
//
// As of yet signerInfos and digestAlgorithms are not parsed, as they are not relevant to
// this system's use of PKCS #7 data.  Version is an integer type, note that PKCS #7 is
// recursive, this second layer of ContentInfo is similar ignored for our degenerate
// usage.  The ExtendedCertificatesAndCertificates type consists of a sequence of choices
// between PKCS #6 extended certificates and x509 certificates.  Any sequence consisting
// of any number of extended certificates is not yet supported in this implementation.
//
// The ContentType Data is simply a raw octet string and is parsed directly into a Go []byte slice.
//
// The ContentType encryptedData is the most complicated and its form can be gathered by
// the go type below.  It essentially contains a raw octet string of encrypted data and an
// algorithm identifier for use in decrypting this data.
package pkcs7

import (
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/asn1"
	"errors"

	cferr "github.com/cloudflare/cfssl/errors"
)

// Types used for asn1 Unmarshaling.

type signedData struct {
	Version          int
	DigestAlgorithms asn1.RawValue
	ContentInfo      asn1.RawValue
	Certificates     asn1.RawValue `asn1:"optional" asn1:"tag:0"`
	Crls             asn1.RawValue `asn1:"optional"`
	SignerInfos      asn1.RawValue
}

type initPKCS7 struct {
	Raw         asn1.RawContent
	ContentType asn1.ObjectIdentifier
	Content     asn1.RawValue `asn1:"tag:0,explicit,optional"`
}

// Object identifier strings of the three implemented PKCS7 types.
const (
	ObjIDData          = "1.2.840.113549.1.7.1"
	ObjIDSignedData    = "1.2.840.113549.1.7.2"
	ObjIDEncryptedData = "1.2.840.113549.1.7.6"
)

// PKCS7 represents the ASN1 PKCS #7 Content type.  It contains one of three
// possible types of Content objects, as denoted by the object identifier in
// the ContentInfo field, the other two being nil.  SignedData
// is the degenerate SignedData Content info without signature used
// to hold certificates and crls.  Data is raw bytes, and EncryptedData
// is as defined in PKCS #7 standard.
type PKCS7 struct {
	Raw         asn1.RawContent
	ContentInfo string
	Content     Content
}

// Content implements three of the six possible PKCS7 data types.  Only one is non-nil.
type Content struct {
	Data          []byte
	SignedData    SignedData
	EncryptedData EncryptedData
}

// SignedData defines the typical carrier of certificates and crls.
type SignedData struct {
	Raw          asn1.RawContent
	Version      int
	Certificates []*x509.Certificate
	Crl          *pkix.CertificateList
}

// Data contains raw bytes.  Used as a subtype in PKCS12.
type Data struct {
	Bytes []byte
}

// EncryptedData contains encrypted data.  Used as a subtype in PKCS12.
type EncryptedData struct {
	Raw                  asn1.RawContent
	Version              int
	EncryptedContentInfo EncryptedContentInfo
}

// EncryptedContentInfo is a subtype of PKCS7EncryptedData.
type EncryptedContentInfo struct {
	Raw                        asn1.RawContent
	ContentType                asn1.ObjectIdentifier
	ContentEncryptionAlgorithm pkix.AlgorithmIdentifier
	EncryptedContent           []byte `asn1:"tag:0,optional"`
}

// ParsePKCS7 attempts to parse the DER encoded bytes of a
// PKCS7 structure.
func ParsePKCS7(raw []byte) (msg *PKCS7, err error) {

	var pkcs7 initPKCS7
	_, err = asn1.Unmarshal(raw, &pkcs7)
	if err != nil {
		return nil, cferr.Wrap(cferr.CertificateError, cferr.ParseFailed, err)
	}

	msg = new(PKCS7)
	msg.Raw = pkcs7.Raw
	msg.ContentInfo = pkcs7.ContentType.String()
	switch {
	case msg.ContentInfo == ObjIDData:
		msg.ContentInfo = "Data"
		_, err = asn1.Unmarshal(pkcs7.Content.Bytes, &msg.Content.Data)
		if err != nil {
			return nil, cferr.Wrap(cferr.CertificateError, cferr.ParseFailed, err)
		}
	case msg.ContentInfo == ObjIDSignedData:
		msg.ContentInfo = "SignedData"
		var signedData signedData
		_, err = asn1.Unmarshal(pkcs7.Content.Bytes, &signedData)
		if err != nil {
			return nil, cferr.Wrap(cferr.CertificateError, cferr.ParseFailed, err)
		}
		if len(signedData.Certificates.Bytes) != 0 {
			msg.Content.SignedData.Certificates, err = x509.ParseCertificates(signedData.Certificates.Bytes)
			if err != nil {
				return nil, cferr.Wrap(cferr.CertificateError, cferr.ParseFailed, err)
			}
		}
		if len(signedData.Crls.Bytes) != 0 {
			msg.Content.SignedData.Crl, err = x509.ParseDERCRL(signedData.Crls.Bytes)
			if err != nil {
				return nil, cferr.Wrap(cferr.CertificateError, cferr.ParseFailed, err)
			}
		}
		msg.Content.SignedData.Version = signedData.Version
		msg.Content.SignedData.Raw = pkcs7.Content.Bytes
	case msg.ContentInfo == ObjIDEncryptedData:
		msg.ContentInfo = "EncryptedData"
		var encryptedData EncryptedData
		_, err = asn1.Unmarshal(pkcs7.Content.Bytes, &encryptedData)
		if err != nil {
			return nil, cferr.Wrap(cferr.CertificateError, cferr.ParseFailed, err)
		}
		if encryptedData.Version != 0 {
			return nil, cferr.Wrap(cferr.CertificateError, cferr.ParseFailed, errors.New("Only support for PKCS #7 encryptedData version 0"))
		}
		msg.Content.EncryptedData = encryptedData

	default:
		return nil, cferr.Wrap(cferr.CertificateError, cferr.ParseFailed, errors.New("Attempt to parse PKCS# 7 Content not of type data, signed data or encrypted data"))
	}

	return msg, nil

}
