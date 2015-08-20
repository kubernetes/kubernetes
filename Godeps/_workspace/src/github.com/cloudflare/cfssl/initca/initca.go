// Package initca contains code to initialise a certificate authority,
// generating a new root key and certificate.
package initca

import (
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"encoding/pem"
	"errors"
	"io/ioutil"
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
	if req.CA != nil {
		if req.CA.Expiry != "" {
			CAPolicy.Default.ExpiryString = req.CA.Expiry
			CAPolicy.Default.Expiry, err = time.ParseDuration(req.CA.Expiry)
		}

		if req.CA.PathLength != 0 {
			signer.MaxPathLen = req.CA.PathLength
		}
	}

	privData, err := ioutil.ReadFile(keyFile)
	if err != nil {
		return nil, nil, err
	}

	priv, err := helpers.ParsePrivateKeyPEM(privData)
	if err != nil {
		return nil, nil, err
	}

	var sigAlgo x509.SignatureAlgorithm
	switch priv := priv.(type) {
	case *rsa.PrivateKey:
		bitLength := priv.PublicKey.N.BitLen()
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
	case *ecdsa.PrivateKey:
		switch priv.Curve {
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
		DNSNames:           req.Hosts,
	}

	csrPEM, err = x509.CreateCertificateRequest(rand.Reader, &tpl, priv)
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

// CAPolicy contains the CA issuing policy as default policy.
var CAPolicy = &config.Signing{
	Default: &config.SigningProfile{
		Usage:        []string{"cert sign", "crl sign"},
		ExpiryString: "43800h",
		Expiry:       5 * helpers.OneYear,
		CA:           true,
	},
}
