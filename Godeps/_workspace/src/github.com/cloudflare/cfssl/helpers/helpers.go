// Package helpers implements utility functionality common to many
// CFSSL packages.
package helpers

import (
	"bytes"
	"crypto"
	"crypto/ecdsa"
	"crypto/rsa"
	"crypto/x509"
	"encoding/pem"
	"errors"
	//"fmt"
	"strings"
	"time"

	"github.com/cloudflare/cfssl/crypto/pkcs12"
	"github.com/cloudflare/cfssl/crypto/pkcs7"
	cferr "github.com/cloudflare/cfssl/errors"
	"github.com/cloudflare/cfssl/helpers/derhelpers"
	"github.com/cloudflare/cfssl/log"
)

// OneYear is a time.Duration representing a year's worth of seconds.
const OneYear = 8760 * time.Hour

// OneDay is a time.Duration representing a day's worth of seconds.
const OneDay = 24 * time.Hour

// InclusiveDate returns the time.Time representation of a date - 1
// nanosecond. This allows time.After to be used inclusively.
func InclusiveDate(year int, month time.Month, day int) time.Time {
	return time.Date(year, month, day, 0, 0, 0, 0, time.UTC).Add(-1 * time.Nanosecond)
}

// Jul2012 is the July 2012 CAB Forum deadline for when CAs must stop
// issuing certificates valid for more than 5 years.
var Jul2012 = InclusiveDate(2012, time.July, 01)

// April2015 is the April 2015 CAB Forum deadline for when CAs must stop
// issuing certificates valid for more than 39 months.
var Apr2015 = InclusiveDate(2015, time.April, 01)

// KeyLength returns the bit size of ECDSA or RSA PublicKey
func KeyLength(key interface{}) int {
	if key == nil {
		return 0
	}
	if ecdsaKey, ok := key.(*ecdsa.PublicKey); ok {
		return ecdsaKey.Curve.Params().BitSize
	} else if rsaKey, ok := key.(*rsa.PublicKey); ok {
		return rsaKey.N.BitLen()
	}

	return 0
}

// ExpiryTime returns the time when the certificate chain is expired.
func ExpiryTime(chain []*x509.Certificate) *time.Time {
	if len(chain) == 0 {
		return nil
	}
	notAfter := chain[0].NotAfter
	for _, cert := range chain {
		if cert.NotAfter.Before(notAfter) {
			notAfter = cert.NotAfter
		}
	}
	return &notAfter
}

// MonthsValid returns the number of months for which a certificate is valid.
func MonthsValid(c *x509.Certificate) int {
	issued := c.NotBefore
	expiry := c.NotAfter
	years := (expiry.Year() - issued.Year())
	months := years*12 + int(expiry.Month()) - int(issued.Month())

	// Round up if valid for less than a full month
	if expiry.Day() > issued.Day() {
		months++
	}
	return months
}

// ValidExpiry determines if a certificate is valid for an acceptable
// length of time per the CA/Browser Forum baseline requirements.
// See https://cabforum.org/wp-content/uploads/CAB-Forum-BR-1.3.0.pdf
func ValidExpiry(c *x509.Certificate) bool {
	issued := c.NotBefore

	var maxMonths int
	switch {
	case issued.After(Apr2015):
		maxMonths = 39
	case issued.After(Jul2012):
		maxMonths = 60
	case issued.Before(Jul2012):
		maxMonths = 120
	}

	if MonthsValid(c) > maxMonths {
		return false
	}
	return true
}

// SignatureString returns the TLS signature string corresponding to
// an X509 signature algorithm.
func SignatureString(alg x509.SignatureAlgorithm) string {
	switch alg {
	case x509.MD2WithRSA:
		return "MD2WithRSA"
	case x509.MD5WithRSA:
		return "MD5WithRSA"
	case x509.SHA1WithRSA:
		return "SHA1WithRSA"
	case x509.SHA256WithRSA:
		return "SHA256WithRSA"
	case x509.SHA384WithRSA:
		return "SHA384WithRSA"
	case x509.SHA512WithRSA:
		return "SHA512WithRSA"
	case x509.DSAWithSHA1:
		return "DSAWithSHA1"
	case x509.DSAWithSHA256:
		return "DSAWithSHA256"
	case x509.ECDSAWithSHA1:
		return "ECDSAWithSHA1"
	case x509.ECDSAWithSHA256:
		return "ECDSAWithSHA256"
	case x509.ECDSAWithSHA384:
		return "ECDSAWithSHA384"
	case x509.ECDSAWithSHA512:
		return "ECDSAWithSHA512"
	default:
		return "Unknown Signature"
	}
}

// HashAlgoString returns the hash algorithm name contains in the signature
// method.
func HashAlgoString(alg x509.SignatureAlgorithm) string {
	switch alg {
	case x509.MD2WithRSA:
		return "MD2"
	case x509.MD5WithRSA:
		return "MD5"
	case x509.SHA1WithRSA:
		return "SHA1"
	case x509.SHA256WithRSA:
		return "SHA256"
	case x509.SHA384WithRSA:
		return "SHA384"
	case x509.SHA512WithRSA:
		return "SHA512"
	case x509.DSAWithSHA1:
		return "SHA1"
	case x509.DSAWithSHA256:
		return "SHA256"
	case x509.ECDSAWithSHA1:
		return "SHA1"
	case x509.ECDSAWithSHA256:
		return "SHA256"
	case x509.ECDSAWithSHA384:
		return "SHA384"
	case x509.ECDSAWithSHA512:
		return "SHA512"
	default:
		return "Unknown Hash Algorithm"
	}
}

// ParseCertificatesPEM parses a sequence of PEM-encoded certificate and returns them,
// can handle PEM encoded PKCS #7 structures.
func ParseCertificatesPEM(certsPEM []byte) ([]*x509.Certificate, error) {
	var certs []*x509.Certificate
	var err error
	certsPEM = bytes.TrimSpace(certsPEM)
	for len(certsPEM) > 0 {
		var cert []*x509.Certificate
		cert, certsPEM, err = ParseOneCertificateFromPEM(certsPEM)
		if err != nil {

			return nil, cferr.New(cferr.CertificateError, cferr.ParseFailed)
		} else if cert == nil {
			break
		}

		certs = append(certs, cert...)
	}
	if len(certsPEM) > 0 {
		return nil, cferr.New(cferr.CertificateError, cferr.DecodeFailed)
	}
	return certs, nil
}

// ParseCertificatesDER parses a DER encoding of a certificate object and possibly private key,
// either PKCS #7, PKCS #12, or raw x509.
func ParseCertificatesDER(certsDER []byte, password string) ([]*x509.Certificate, crypto.Signer, error) {
	var certs []*x509.Certificate
	var key crypto.Signer
	certsDER = bytes.TrimSpace(certsDER)
	pkcs7data, err := pkcs7.ParsePKCS7(certsDER)
	if err != nil {
		pkcs12data, err := pkcs12.ParsePKCS12(certsDER, []byte(password))
		if err != nil {
			certs, err = x509.ParseCertificates(certsDER)
			if err != nil {
				return nil, nil, cferr.New(cferr.CertificateError, cferr.DecodeFailed)
			}
		} else {
			key = pkcs12data.PrivateKey
			certs = pkcs12data.Certificates
		}
	} else {
		if pkcs7data.ContentInfo != "SignedData" {
			return nil, nil, cferr.Wrap(cferr.CertificateError, cferr.DecodeFailed, errors.New("can only extract certificates from signed data content info"))
		}
		certs = pkcs7data.Content.SignedData.Certificates
	}
	if certs == nil {
		return nil, key, cferr.New(cferr.CertificateError, cferr.DecodeFailed)
	}
	return certs, key, nil
}

// ParseSelfSignedCertificatePEM parses a PEM-encoded certificate and check if it is self-signed.
func ParseSelfSignedCertificatePEM(certPEM []byte) (*x509.Certificate, error) {
	cert, err := ParseCertificatePEM(certPEM)
	if err != nil {
		return nil, err
	}
	if err := cert.CheckSignature(cert.SignatureAlgorithm, cert.RawTBSCertificate, cert.Signature); err != nil {
		return nil, cferr.Wrap(cferr.CertificateError, cferr.VerifyFailed, err)
	}
	return cert, nil
}

// ParseCertificatePEM parses and returns a PEM-encoded certificate,
// can handle PEM encoded PKCS #7 structures.
func ParseCertificatePEM(certPEM []byte) (*x509.Certificate, error) {
	certPEM = bytes.TrimSpace(certPEM)
	cert, rest, err := ParseOneCertificateFromPEM(certPEM)
	if err != nil {
		// Log the actual parsing error but throw a default parse error message.
		log.Debugf("Certificate parsing error: %v", err)
		return nil, cferr.New(cferr.CertificateError, cferr.ParseFailed)
	} else if cert == nil {
		return nil, cferr.New(cferr.CertificateError, cferr.DecodeFailed)
	} else if len(rest) > 0 {
		return nil, cferr.Wrap(cferr.CertificateError, cferr.ParseFailed, errors.New("the PEM file should contain only one object"))
	} else if len(cert) > 1 {
		return nil, cferr.Wrap(cferr.CertificateError, cferr.ParseFailed, errors.New("the PKCS7 object in the PEM file should contain only one certificate"))
	}
	return cert[0], nil
}

// ParseOneCertificateFromPEM attempts to parse one PEM encoded certificate object,
// either a raw x509 certificate or a PKCS #7 structure possibly containing
// multiple certificates, from the top of certsPEM, which itself may
// contain multiple PEM encoded certificate objects.
func ParseOneCertificateFromPEM(certsPEM []byte) ([]*x509.Certificate, []byte, error) {

	block, rest := pem.Decode(certsPEM)
	if block == nil {
		return nil, rest, nil
	}

	cert, err := x509.ParseCertificate(block.Bytes)
	if err != nil {
		pkcs7data, err := pkcs7.ParsePKCS7(block.Bytes)
		if err != nil {
			return nil, rest, err
		}
		if pkcs7data.ContentInfo != "SignedData" {
			return nil, rest, errors.New("only PKCS #7 Signed Data Content Info supported for certificate parsing")
		}
		certs := pkcs7data.Content.SignedData.Certificates
		if certs == nil {
			return nil, rest, errors.New("PKCS #7 structure contains no certificates")
		}
		return certs, rest, nil
	}
	var certs = []*x509.Certificate{cert}
	return certs, rest, nil
}

// ParsePrivateKeyPEM parses and returns a PEM-encoded private
// key. The private key may be either an unencrypted PKCS#8, PKCS#1,
// or elliptic private key.
func ParsePrivateKeyPEM(keyPEM []byte) (key crypto.Signer, err error) {
	keyDER, err := GetKeyDERFromPEM(keyPEM)
	if err != nil {
		return nil, err
	}

	return derhelpers.ParsePrivateKeyDER(keyDER)
}

// GetKeyDERFromPEM parses a PEM-encoded private key and returns DER-format key bytes.
func GetKeyDERFromPEM(in []byte) ([]byte, error) {
	keyDER, _ := pem.Decode(in)
	if keyDER != nil {
		if procType, ok := keyDER.Headers["Proc-Type"]; ok {
			if strings.Contains(procType, "ENCRYPTED") {
				return nil, cferr.New(cferr.PrivateKeyError, cferr.Encrypted)
			}
		}
		return keyDER.Bytes, nil
	}

	return nil, cferr.New(cferr.PrivateKeyError, cferr.DecodeFailed)
}
