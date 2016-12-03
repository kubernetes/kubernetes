// Package initca contains code to initialise a certificate authority,
// generating a new root key and certificate.
package initca

import (
	"crypto"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"encoding/pem"
	"errors"
	"io/ioutil"
	"net"
	"time"

	"github.com/cloudflare/cfssl/config"
	"github.com/cloudflare/cfssl/csr"
	cferr "github.com/cloudflare/cfssl/errors"
	"github.com/cloudflare/cfssl/helpers"
	"github.com/cloudflare/cfssl/log"
	"github.com/cloudflare/cfssl/signer"
	"github.com/cloudflare/cfssl/signer/local"
)

// validator contains the default validation logic for certificate
// authority certificates. The only requirement here is that the
// certificate have a non-empty subject field.
func validator(req *csr.CertificateRequest) error {
	if req.CN != "" {
		return nil
	}

	if len(req.Names) == 0 {
		return cferr.Wrap(cferr.PolicyError, cferr.InvalidRequest, errors.New("missing subject information"))
	}

	for i := range req.Names {
		if csr.IsNameEmpty(req.Names[i]) {
			return cferr.Wrap(cferr.PolicyError, cferr.InvalidRequest, errors.New("missing subject information"))
		}
	}

	return nil
}

// New creates a new root certificate from the certificate request.
func New(req *csr.CertificateRequest) (cert, csrPEM, key []byte, err error) {
	if req.CA != nil {
		if req.CA.Expiry != "" {
			CAPolicy.Default.ExpiryString = req.CA.Expiry
			CAPolicy.Default.Expiry, err = time.ParseDuration(req.CA.Expiry)
		}

		if req.CA.PathLength != 0 {
			signer.MaxPathLen = req.CA.PathLength
		}
	}

	g := &csr.Generator{Validator: validator}
	csrPEM, key, err = g.ProcessRequest(req)
	if err != nil {
		log.Errorf("failed to process request: %v", err)
		key = nil
		return
	}

	priv, err := helpers.ParsePrivateKeyPEM(key)
	if err != nil {
		log.Errorf("failed to parse private key: %v", err)
		return
	}

	s, err := local.NewSigner(priv, nil, signer.DefaultSigAlgo(priv), nil)
	if err != nil {
		log.Errorf("failed to create signer: %v", err)
		return
	}
	s.SetPolicy(CAPolicy)

	signReq := signer.SignRequest{Hosts: req.Hosts, Request: string(csrPEM)}
	cert, err = s.Sign(signReq)

	return

}

// NewFromPEM creates a new root certificate from the key file passed in.
func NewFromPEM(req *csr.CertificateRequest, keyFile string) (cert, csrPEM []byte, err error) {
	privData, err := ioutil.ReadFile(keyFile)
	if err != nil {
		return nil, nil, err
	}

	priv, err := helpers.ParsePrivateKeyPEM(privData)
	if err != nil {
		return nil, nil, err
	}

	return NewFromSigner(req, priv)
}

// RenewFromPEM re-creates a root certificate from the CA cert and key
// files. The resulting root certificate will have the input CA certificate
// as the template and have the same expiry length. E.g. the exsiting CA
// is valid for a year from Jan 01 2015 to Jan 01 2016, the renewed certificate
// will be valid from now and expire in one year as well.
func RenewFromPEM(caFile, keyFile string) ([]byte, error) {
	caBytes, err := ioutil.ReadFile(caFile)
	if err != nil {
		return nil, err
	}

	ca, err := helpers.ParseCertificatePEM(caBytes)
	if err != nil {
		return nil, err
	}

	keyBytes, err := ioutil.ReadFile(keyFile)
	if err != nil {
		return nil, err
	}

	key, err := helpers.ParsePrivateKeyPEM(keyBytes)
	if err != nil {
		return nil, err
	}

	return RenewFromSigner(ca, key)

}

// NewFromSigner creates a new root certificate from a crypto.Signer.
func NewFromSigner(req *csr.CertificateRequest, priv crypto.Signer) (cert, csrPEM []byte, err error) {
	if req.CA != nil {
		if req.CA.Expiry != "" {
			CAPolicy.Default.ExpiryString = req.CA.Expiry
			CAPolicy.Default.Expiry, err = time.ParseDuration(req.CA.Expiry)
			if err != nil {
				return nil, nil, err
			}
		}

		if req.CA.PathLength != 0 {
			signer.MaxPathLen = req.CA.PathLength
		}
	}

	var sigAlgo x509.SignatureAlgorithm
	switch pub := priv.Public().(type) {
	case *rsa.PublicKey:
		bitLength := pub.N.BitLen()
		switch {
		case bitLength >= 4096:
			sigAlgo = x509.SHA512WithRSA
		case bitLength >= 3072:
			sigAlgo = x509.SHA384WithRSA
		case bitLength >= 2048:
			sigAlgo = x509.SHA256WithRSA
		default:
			sigAlgo = x509.SHA1WithRSA
		}
	case *ecdsa.PublicKey:
		switch pub.Curve {
		case elliptic.P521():
			sigAlgo = x509.ECDSAWithSHA512
		case elliptic.P384():
			sigAlgo = x509.ECDSAWithSHA384
		case elliptic.P256():
			sigAlgo = x509.ECDSAWithSHA256
		default:
			sigAlgo = x509.ECDSAWithSHA1
		}
	default:
		sigAlgo = x509.UnknownSignatureAlgorithm
	}

	var tpl = x509.CertificateRequest{
		Subject:            req.Name(),
		SignatureAlgorithm: sigAlgo,
	}

	for i := range req.Hosts {
		if ip := net.ParseIP(req.Hosts[i]); ip != nil {
			tpl.IPAddresses = append(tpl.IPAddresses, ip)
		} else {
			tpl.DNSNames = append(tpl.DNSNames, req.Hosts[i])
		}
	}

	return signWithCSR(&tpl, priv)
}

// signWithCSR creates a new root certificate from signing a X509.CertificateRequest
// by a crypto.Signer.
func signWithCSR(tpl *x509.CertificateRequest, priv crypto.Signer) (cert, csrPEM []byte, err error) {
	csrPEM, err = x509.CreateCertificateRequest(rand.Reader, tpl, priv)
	if err != nil {
		log.Errorf("failed to generate a CSR: %v", err)
		// The use of CertificateError was a matter of some
		// debate; it is the one edge case in which a new
		// error category specifically for CSRs might be
		// useful, but it was deemed that one edge case did
		// not a new category justify.
		err = cferr.Wrap(cferr.CertificateError, cferr.BadRequest, err)
		return
	}

	p := &pem.Block{
		Type:  "CERTIFICATE REQUEST",
		Bytes: csrPEM,
	}
	csrPEM = pem.EncodeToMemory(p)

	s, err := local.NewSigner(priv, nil, signer.DefaultSigAlgo(priv), nil)
	if err != nil {
		log.Errorf("failed to create signer: %v", err)
		return
	}
	s.SetPolicy(CAPolicy)

	signReq := signer.SignRequest{Request: string(csrPEM)}
	cert, err = s.Sign(signReq)
	return
}

// RenewFromSigner re-creates a root certificate from the CA cert and crypto.Signer.
// The resulting root certificate will have ca certificate
// as the template and have the same expiry length. E.g. the exsiting CA
// is valid for a year from Jan 01 2015 to Jan 01 2016, the renewed certificate
// will be valid from now and expire in one year as well.
func RenewFromSigner(ca *x509.Certificate, priv crypto.Signer) ([]byte, error) {
	if !ca.IsCA {
		return nil, errors.New("input certificate is not a CA cert")
	}

	// matching certificate public key vs private key
	switch {
	case ca.PublicKeyAlgorithm == x509.RSA:

		var rsaPublicKey *rsa.PublicKey
		var ok bool
		if rsaPublicKey, ok = priv.Public().(*rsa.PublicKey); !ok {
			return nil, cferr.New(cferr.PrivateKeyError, cferr.KeyMismatch)
		}
		if ca.PublicKey.(*rsa.PublicKey).N.Cmp(rsaPublicKey.N) != 0 {
			return nil, cferr.New(cferr.PrivateKeyError, cferr.KeyMismatch)
		}
	case ca.PublicKeyAlgorithm == x509.ECDSA:
		var ecdsaPublicKey *ecdsa.PublicKey
		var ok bool
		if ecdsaPublicKey, ok = priv.Public().(*ecdsa.PublicKey); !ok {
			return nil, cferr.New(cferr.PrivateKeyError, cferr.KeyMismatch)
		}
		if ca.PublicKey.(*ecdsa.PublicKey).X.Cmp(ecdsaPublicKey.X) != 0 {
			return nil, cferr.New(cferr.PrivateKeyError, cferr.KeyMismatch)
		}
	default:
		return nil, cferr.New(cferr.PrivateKeyError, cferr.NotRSAOrECC)
	}

	req := csr.ExtractCertificateRequest(ca)

	cert, _, err := NewFromSigner(req, priv)
	return cert, err

}

// CAPolicy contains the CA issuing policy as default policy.
var CAPolicy = &config.Signing{
	Default: &config.SigningProfile{
		Usage:        []string{"cert sign", "crl sign"},
		ExpiryString: "43800h",
		Expiry:       5 * helpers.OneYear,
		CA:           true,
	},
}
