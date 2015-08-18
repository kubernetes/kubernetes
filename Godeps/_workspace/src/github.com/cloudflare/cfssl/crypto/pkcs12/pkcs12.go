// Package pkcs12 implements a subset of PKCS #12 as described here:
// https://tools.ietf.org/html/rfc7292
//
//
// Much credit to at Microsoft's Azure project https:
// github.com/Azure/go-pkcs12/blob/master/pkcs12.go,
// from which much of the parser code was adapted under the MIT License.
// PKCS #12 is a format used for transferring certificates and private keys.
//
//
// In particular the PFX/P12 structure storing certificates and private keys is parsed into a go structure.
// In almost all cases PKCS #12 stored certificates and private keys are password protected at the time of
// marshaling, and so the parse function in this package takes in a password []byte.  PKCS #12 make extensive
// use of the PKCS #7 standard, and so the PKCS #7 parser is used frequently here.  Although there is
// flexibility in the data a PKCS #12 object can hold, the typical (i.e. openssl generated) form is roughly
// as follows (for more specific details on allowed asn1 structure see the standard)
//
//
//			PFX ->
//				Version int
//				PKCS #7 Data ->
//					PKCS #7 encryptedData ->
//						CertificateBag ->
//							Certificates
//					PKCS #7 Data ->
//						PKCS #8 ShroudedBag ->
//							Private Key
//				MAC Data (Not used here)
//
package pkcs12

import (
	"crypto"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/asn1"
	"errors"

	"github.com/cloudflare/cfssl/crypto/pkcs12/pbkdf"
	"github.com/cloudflare/cfssl/crypto/pkcs7"
	cferr "github.com/cloudflare/cfssl/errors"
	"github.com/cloudflare/cfssl/helpers/derhelpers"
)

const (
	certBagID          = "1.2.840.113549.1.12.10.1.3"
	pkcs8ShroudedBagID = "1.2.840.113549.1.12.10.1.2"
)

// Internal types used for asn1 Unmarshaling
type pfx struct {
	Raw      asn1.RawContent
	Version  int
	AuthSafe asn1.RawValue
	MacData  asn1.RawValue `asn1:"optional"`
}

type contentInfo struct {
	ContentType asn1.ObjectIdentifier
	Content     asn1.RawValue `asn1:"tag:0,explicit,optional"`
}

type safeBag struct {
	ID         asn1.ObjectIdentifier
	Value      asn1.RawValue     `asn1:"tag:0,explicit"`
	Attributes []pkcs12Attribute `asn1:"set,optional"`
}

type certBag struct {
	ID   asn1.ObjectIdentifier
	Data []byte `asn1:"tag:0,explicit"`
}

type pkcs12Attribute struct {
	ID    asn1.ObjectIdentifier
	Value asn1.RawValue `asn1:"set"`
}

type encryptedPrivateKeyInfo struct {
	AlgorithmIdentifier pkix.AlgorithmIdentifier
	EncryptedData       []byte
}

// PKCS12 contains the Data expected in PKCS #12 objects, one or more certificates
// a private key, an integer indicating the version, and the raw content
// of the structure
type PKCS12 struct {
	Version      int
	Certificates []*x509.Certificate
	PrivateKey   crypto.Signer
}

// ParsePKCS12 parses a pkcs12 syntax object
// into a container for a private key, certificate(s), and
// version number
func ParsePKCS12(raw, password []byte) (msg *PKCS12, err error) {
	msg = new(PKCS12)
	password, err = pbkdf.BMPString(password)
	if err != nil {
		return nil, cferr.Wrap(cferr.CertificateError, cferr.ParseFailed, err)
	}
	var Pfx pfx
	_, err = asn1.Unmarshal(raw, &Pfx)
	if err != nil {
		return nil, cferr.Wrap(cferr.CertificateError, cferr.ParseFailed, err)
	}

	if msg.Version = Pfx.Version; msg.Version != 3 {
		return nil, cferr.Wrap(cferr.CertificateError, cferr.ParseFailed, errors.New("Only support for PKCS #12 PFX version 3"))
	}
	authSafe, err := pkcs7.ParsePKCS7(Pfx.AuthSafe.FullBytes)

	if err != nil {
		return nil, err
	}
	if authSafe.ContentInfo != "Data" {
		return nil, cferr.Wrap(cferr.CertificateError, cferr.ParseFailed, errors.New("No support for AuthSafe Format"))
	}
	var authenticatedSafe []asn1.RawValue
	_, err = asn1.Unmarshal(authSafe.Content.Data, &authenticatedSafe)
	if err != nil {
		return nil, cferr.Wrap(cferr.CertificateError, cferr.ParseFailed, err)

	}

	if len(authenticatedSafe) != 2 {
		return nil, cferr.Wrap(cferr.CertificateError, cferr.ParseFailed, errors.New("No support for AuthSafe Format"))
	}
	var bags []safeBag
	bags, err = getBags(authenticatedSafe, password)
	if err != nil {
		return nil, err
	}
	if len(bags) > 2 || bags == nil {
		return nil, cferr.Wrap(cferr.CertificateError, cferr.ParseFailed, errors.New("No support for AuthSafe Format"))
	}

	certs, pkey, err := parseBags(bags, password)
	if err != nil {
		return nil, cferr.Wrap(cferr.CertificateError, cferr.ParseFailed, err)
	}
	msg.Certificates = certs
	msg.PrivateKey = pkey
	return

}

// Given a slice of PKCS #7 content infos containing PKCS #12 Safe Bag Data,
// getBags returns those Safe Bags.
func getBags(authenticatedSafe []asn1.RawValue, password []byte) (bags []safeBag, err error) {
	for _, contentInfo := range authenticatedSafe {

		var safeContents []safeBag
		bagContainer, err := pkcs7.ParsePKCS7(contentInfo.FullBytes)
		if err != nil {
			return nil, err
		}
		switch {
		case bagContainer.ContentInfo == "Data":
			if _, err = asn1.Unmarshal(bagContainer.Content.Data, &safeContents); err != nil {
				return nil, cferr.Wrap(cferr.CertificateError, cferr.ParseFailed, err)
			}
		case bagContainer.ContentInfo == "EncryptedData":
			data, err := decrypt(bagContainer.Content.EncryptedData.EncryptedContentInfo.ContentEncryptionAlgorithm,
				bagContainer.Content.EncryptedData.EncryptedContentInfo.EncryptedContent, password)
			if err != nil {
				return nil, cferr.Wrap(cferr.CertificateError, cferr.ParseFailed, err)
			}
			if _, err = asn1.Unmarshal(data, &safeContents); err != nil {
				return nil, cferr.Wrap(cferr.CertificateError, cferr.ParseFailed, err)
			}
		default:
			return nil, cferr.Wrap(cferr.CertificateError, cferr.ParseFailed, errors.New("Only support for bags encoded in Data and EncryptedData types"))
		}

		bags = append(bags, safeContents...)
	}
	return bags, nil

}

// Take in either two or one safeBags and return the certificates and or
// Private key within the bags
func parseBags(bags []safeBag, password []byte) (certs []*x509.Certificate, key crypto.Signer, err error) {
	for _, bag := range bags {
		bagid := bag.ID.String()
		switch bagid {
		case certBagID:
			var CertBag certBag
			if _, err = asn1.Unmarshal(bag.Value.Bytes, &CertBag); err != nil {
				return nil, nil, cferr.Wrap(cferr.CertificateError, cferr.ParseFailed, err)
			}
			certs, err = x509.ParseCertificates(CertBag.Data)

			if err != nil {
				return nil, nil, cferr.Wrap(cferr.CertificateError, cferr.ParseFailed, err)
			}

		case pkcs8ShroudedBagID:
			var pkinfo encryptedPrivateKeyInfo
			if _, err := asn1.Unmarshal(bag.Value.Bytes, &pkinfo); err != nil {
				return nil, nil, cferr.Wrap(cferr.CertificateError, cferr.ParseFailed, err)
			}
			pkDecrypted, err := decrypt(pkinfo.AlgorithmIdentifier, pkinfo.EncryptedData, password)
			if err != nil {
				return nil, nil, cferr.Wrap(cferr.CertificateError, cferr.ParseFailed, err)
			}
			// Checking if private key data has been properly decoded
			var rv asn1.RawValue
			if _, err = asn1.Unmarshal(pkDecrypted, &rv); err != nil {
				return nil, nil, cferr.Wrap(cferr.CertificateError, cferr.ParseFailed, err)
			}
			if key, err = derhelpers.ParsePrivateKeyDER(pkDecrypted); err != nil {
				return nil, nil, err
			}

		default:
			return nil, nil, cferr.Wrap(cferr.CertificateError, cferr.ParseFailed, errors.New("Only support for certificate bags and PKCS #8 Shrouded Bags"))
		}
	}
	return certs, key, nil
}
