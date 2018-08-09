// Package initca contains code to initialise a certificate authority,
// generating a new root key and certificate.
package initca

import (
	"crypto"
	"crypto/ecdsa"
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"encoding/pem"
	"errors"
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
	policy := CAPolicy()
	if req.CA != nil {
		if req.CA.Expiry != "" {
			policy.Default.ExpiryString = req.CA.Expiry
			policy.Default.Expiry, err = time.ParseDuration(req.CA.Expiry)
			if err != nil {
				return
			}
		}

		if req.CA.Backdate != "" {
			policy.Default.Backdate, err = time.ParseDuration(req.CA.Backdate)
			if err != nil {
				return
			}
		}

		policy.Default.CAConstraint.MaxPathLen = req.CA.PathLength
		if req.CA.PathLength != 0 && req.CA.PathLenZero {
			log.Infof("ignore invalid 'pathlenzero' value")
		} else {
			policy.Default.CAConstraint.MaxPathLenZero = req.CA.PathLenZero
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

	s, err := local.NewSigner(priv, nil, signer.DefaultSigAlgo(priv), policy)
	if err != nil {
		log.Errorf("failed to create signer: %v", err)
		return
	}

	signReq := signer.SignRequest{Hosts: req.Hosts, Request: string(csrPEM)}
	cert, err = s.Sign(signReq)

	return

}

// NewFromPEM creates a new root certificate from the key file passed in.
func NewFromPEM(req *csr.CertificateRequest, keyFile string) (cert, csrPEM []byte, err error) {
	privData, err := helpers.ReadBytes(keyFile)
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
// as the template and have the same expiry length. E.g. the existing CA
// is valid for a year from Jan 01 2015 to Jan 01 2016, the renewed certificate
// will be valid from now and expire in one year as well.
func RenewFromPEM(caFile, keyFile string) ([]byte, error) {
	caBytes, err := helpers.ReadBytes(caFile)
	if err != nil {
		return nil, err
	}

	ca, err := helpers.ParseCertificatePEM(caBytes)
	if err != nil {
		return nil, err
	}

	keyBytes, err := helpers.ReadBytes(keyFile)
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
	policy := CAPolicy()
	if req.CA != nil {
		if req.CA.Expiry != "" {
			policy.Default.ExpiryString = req.CA.Expiry
			policy.Default.Expiry, err = time.ParseDuration(req.CA.Expiry)
			if err != nil {
				return nil, nil, err
			}
		}

		policy.Default.CAConstraint.MaxPathLen = req.CA.PathLength
		if req.CA.PathLength != 0 && req.CA.PathLenZero == true {
			log.Infof("ignore invalid 'pathlenzero' value")
		} else {
			policy.Default.CAConstraint.MaxPathLenZero = req.CA.PathLenZero
		}
	}

	csrPEM, err = csr.Generate(priv, req)
	if err != nil {
		return nil, nil, err
	}

	s, err := local.NewSigner(priv, nil, signer.DefaultSigAlgo(priv), policy)
	if err != nil {
		log.Errorf("failed to create signer: %v", err)
		return
	}

	signReq := signer.SignRequest{Request: string(csrPEM)}
	cert, err = s.Sign(signReq)
	return
}

// RenewFromSigner re-creates a root certificate from the CA cert and crypto.Signer.
// The resulting root certificate will have ca certificate
// as the template and have the same expiry length. E.g. the existing CA
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
var CAPolicy = func() *config.Signing {
	return &config.Signing{
		Default: &config.SigningProfile{
			Usage:        []string{"cert sign", "crl sign"},
			ExpiryString: "43800h",
			Expiry:       5 * helpers.OneYear,
			CAConstraint: config.CAConstraint{IsCA: true},
		},
	}
}

// Update copies the CA certificate, updates the NotBefore and
// NotAfter fields, and then re-signs the certificate.
func Update(ca *x509.Certificate, priv crypto.Signer) (cert []byte, err error) {
	copy, err := x509.ParseCertificate(ca.Raw)
	if err != nil {
		return
	}

	validity := ca.NotAfter.Sub(ca.NotBefore)
	copy.NotBefore = time.Now().Round(time.Minute).Add(-5 * time.Minute)
	copy.NotAfter = copy.NotBefore.Add(validity)
	cert, err = x509.CreateCertificate(rand.Reader, copy, copy, priv.Public(), priv)
	if err != nil {
		return
	}

	cert = pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: cert})
	return
}
