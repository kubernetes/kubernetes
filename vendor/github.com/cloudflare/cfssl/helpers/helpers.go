// Package helpers implements utility functionality common to many
// CFSSL packages.
package helpers

import (
	"bytes"
	"crypto"
	"crypto/ecdsa"
	"crypto/rsa"
	"crypto/x509"
	"encoding/asn1"
	"encoding/pem"
	"errors"
	"io/ioutil"
	"math/big"

	"strings"
	"time"

	"github.com/cloudflare/cfssl/crypto/pkcs7"
	cferr "github.com/cloudflare/cfssl/errors"
	"github.com/cloudflare/cfssl/helpers/derhelpers"
	"github.com/cloudflare/cfssl/log"
	"golang.org/x/crypto/pkcs12"
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

// Apr2015 is the April 2015 CAB Forum deadline for when CAs must stop
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
func ExpiryTime(chain []*x509.Certificate) (notAfter time.Time) {
	if len(chain) == 0 {
		return
	}

	notAfter = chain[0].NotAfter
	for _, cert := range chain {
		if notAfter.After(cert.NotAfter) {
			notAfter = cert.NotAfter
		}
	}
	return
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

// EncodeCertificatesPEM encodes a number of x509 certficates to PEM
func EncodeCertificatesPEM(certs []*x509.Certificate) []byte {
	var buffer bytes.Buffer
	for _, cert := range certs {
		pem.Encode(&buffer, &pem.Block{
			Type:  "CERTIFICATE",
			Bytes: cert.Raw,
		})
	}

	return buffer.Bytes()
}

// EncodeCertificatePEM encodes a single x509 certficates to PEM
func EncodeCertificatePEM(cert *x509.Certificate) []byte {
	return EncodeCertificatesPEM([]*x509.Certificate{cert})
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
func ParseCertificatesDER(certsDER []byte, password string) (certs []*x509.Certificate, key crypto.Signer, err error) {
	certsDER = bytes.TrimSpace(certsDER)
	pkcs7data, err := pkcs7.ParsePKCS7(certsDER)
	if err != nil {
		var pkcs12data interface{}
		certs = make([]*x509.Certificate, 1)
		pkcs12data, certs[0], err = pkcs12.Decode(certsDER, password)
		if err != nil {
			certs, err = x509.ParseCertificates(certsDER)
			if err != nil {
				return nil, nil, cferr.New(cferr.CertificateError, cferr.DecodeFailed)
			}
		} else {
			key = pkcs12data.(crypto.Signer)
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

// LoadPEMCertPool loads a pool of PEM certificates from file.
func LoadPEMCertPool(certsFile string) (*x509.CertPool, error) {
	pemCerts, err := ioutil.ReadFile(certsFile)
	if err != nil {
		return nil, err
	}

	certPool := x509.NewCertPool()
	if !certPool.AppendCertsFromPEM(pemCerts) {
		return nil, errors.New("failed to load cert pool")
	}

	return certPool, nil
}

// ParsePrivateKeyPEM parses and returns a PEM-encoded private
// key. The private key may be either an unencrypted PKCS#8, PKCS#1,
// or elliptic private key.
func ParsePrivateKeyPEM(keyPEM []byte) (key crypto.Signer, err error) {
	return ParsePrivateKeyPEMWithPassword(keyPEM, nil)
}

// ParsePrivateKeyPEMWithPassword parses and returns a PEM-encoded private
// key. The private key may be a potentially encrypted PKCS#8, PKCS#1,
// or elliptic private key.
func ParsePrivateKeyPEMWithPassword(keyPEM []byte, password []byte) (key crypto.Signer, err error) {
	keyDER, err := GetKeyDERFromPEM(keyPEM, password)
	if err != nil {
		return nil, err
	}

	return derhelpers.ParsePrivateKeyDER(keyDER)
}

// GetKeyDERFromPEM parses a PEM-encoded private key and returns DER-format key bytes.
func GetKeyDERFromPEM(in []byte, password []byte) ([]byte, error) {
	keyDER, _ := pem.Decode(in)
	if keyDER != nil {
		if procType, ok := keyDER.Headers["Proc-Type"]; ok {
			if strings.Contains(procType, "ENCRYPTED") {
				if password != nil {
					return x509.DecryptPEMBlock(keyDER, password)
				}
				return nil, cferr.New(cferr.PrivateKeyError, cferr.Encrypted)
			}
		}
		return keyDER.Bytes, nil
	}

	return nil, cferr.New(cferr.PrivateKeyError, cferr.DecodeFailed)
}

// CheckSignature verifies a signature made by the key on a CSR, such
// as on the CSR itself.
func CheckSignature(csr *x509.CertificateRequest, algo x509.SignatureAlgorithm, signed, signature []byte) error {
	var hashType crypto.Hash

	switch algo {
	case x509.SHA1WithRSA, x509.ECDSAWithSHA1:
		hashType = crypto.SHA1
	case x509.SHA256WithRSA, x509.ECDSAWithSHA256:
		hashType = crypto.SHA256
	case x509.SHA384WithRSA, x509.ECDSAWithSHA384:
		hashType = crypto.SHA384
	case x509.SHA512WithRSA, x509.ECDSAWithSHA512:
		hashType = crypto.SHA512
	default:
		return x509.ErrUnsupportedAlgorithm
	}

	if !hashType.Available() {
		return x509.ErrUnsupportedAlgorithm
	}
	h := hashType.New()

	h.Write(signed)
	digest := h.Sum(nil)

	switch pub := csr.PublicKey.(type) {
	case *rsa.PublicKey:
		return rsa.VerifyPKCS1v15(pub, hashType, digest, signature)
	case *ecdsa.PublicKey:
		ecdsaSig := new(struct{ R, S *big.Int })
		if _, err := asn1.Unmarshal(signature, ecdsaSig); err != nil {
			return err
		}
		if ecdsaSig.R.Sign() <= 0 || ecdsaSig.S.Sign() <= 0 {
			return errors.New("x509: ECDSA signature contained zero or negative values")
		}
		if !ecdsa.Verify(pub, digest, ecdsaSig.R, ecdsaSig.S) {
			return errors.New("x509: ECDSA verification failure")
		}
		return nil
	}
	return x509.ErrUnsupportedAlgorithm
}

// ParseCSR parses a PEM- or DER-encoded PKCS #10 certificate signing request.
func ParseCSR(in []byte) (csr *x509.CertificateRequest, rest []byte, err error) {
	in = bytes.TrimSpace(in)
	p, rest := pem.Decode(in)
	if p != nil {
		if p.Type != "CERTIFICATE REQUEST" {
			return nil, rest, cferr.New(cferr.CSRError, cferr.BadRequest)
		}

		csr, err = x509.ParseCertificateRequest(p.Bytes)
	} else {
		csr, err = x509.ParseCertificateRequest(in)
	}

	if err != nil {
		return nil, rest, err
	}

	err = CheckSignature(csr, csr.SignatureAlgorithm, csr.RawTBSCertificateRequest, csr.Signature)
	if err != nil {
		return nil, rest, err
	}

	return csr, rest, nil
}

// ParseCSRPEM parses a PEM-encoded certificiate signing request.
// It does not check the signature. This is useful for dumping data from a CSR
// locally.
func ParseCSRPEM(csrPEM []byte) (*x509.CertificateRequest, error) {
	block, _ := pem.Decode([]byte(csrPEM))
	der := block.Bytes
	csrObject, err := x509.ParseCertificateRequest(der)

	if err != nil {
		return nil, err
	}

	return csrObject, nil
}

// SignerAlgo returns an X.509 signature algorithm corresponding to
// the crypto.Hash provided from a crypto.Signer.
func SignerAlgo(priv crypto.Signer, h crypto.Hash) x509.SignatureAlgorithm {
	switch priv.Public().(type) {
	case *rsa.PublicKey:
		switch h {
		case crypto.SHA512:
			return x509.SHA512WithRSA
		case crypto.SHA384:
			return x509.SHA384WithRSA
		case crypto.SHA256:
			return x509.SHA256WithRSA
		default:
			return x509.SHA1WithRSA
		}
	case *ecdsa.PublicKey:
		switch h {
		case crypto.SHA512:
			return x509.ECDSAWithSHA512
		case crypto.SHA384:
			return x509.ECDSAWithSHA384
		case crypto.SHA256:
			return x509.ECDSAWithSHA256
		default:
			return x509.ECDSAWithSHA1
		}
	default:
		return x509.UnknownSignatureAlgorithm
	}
}
