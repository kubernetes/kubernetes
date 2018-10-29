// Package selfsign implements certificate selfsigning. This is very
// dangerous and should never be used in production.
package selfsign

import (
	"crypto"
	"crypto/rand"
	"crypto/sha1"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/asn1"
	"encoding/pem"
	"errors"
	"math"
	"math/big"
	"time"

	"github.com/cloudflare/cfssl/config"
	cferr "github.com/cloudflare/cfssl/errors"
	"github.com/cloudflare/cfssl/signer"
)

const threeMonths = 2190 * time.Hour

// parseCertificateRequest takes an incoming certificate request and
// builds a certificate template from it.
func parseCertificateRequest(priv crypto.Signer, csrBytes []byte) (template *x509.Certificate, err error) {

	csr, err := x509.ParseCertificateRequest(csrBytes)
	if err != nil {
		err = cferr.Wrap(cferr.CSRError, cferr.ParseFailed, err)
		return
	}

	csr.CheckSignature()
	if err != nil {
		err = cferr.Wrap(cferr.CSRError, cferr.KeyMismatch, err)
		return
	}

	template = &x509.Certificate{
		Subject:            csr.Subject,
		PublicKeyAlgorithm: csr.PublicKeyAlgorithm,
		PublicKey:          csr.PublicKey,
		SignatureAlgorithm: signer.DefaultSigAlgo(priv),
	}

	return
}

type subjectPublicKeyInfo struct {
	Algorithm        pkix.AlgorithmIdentifier
	SubjectPublicKey asn1.BitString
}

// Sign creates a new self-signed certificate.
func Sign(priv crypto.Signer, csrPEM []byte, profile *config.SigningProfile) ([]byte, error) {
	if profile == nil {
		return nil, cferr.Wrap(cferr.PolicyError, cferr.InvalidPolicy, errors.New("no profile for self-signing"))
	}

	p, _ := pem.Decode(csrPEM)
	if p == nil || p.Type != "CERTIFICATE REQUEST" {
		return nil, cferr.New(cferr.CSRError, cferr.BadRequest)
	}

	template, err := parseCertificateRequest(priv, p.Bytes)
	if err != nil {
		return nil, err
	}

	pub := template.PublicKey
	encodedpub, err := x509.MarshalPKIXPublicKey(pub)
	if err != nil {
		return nil, err
	}
	var subPKI subjectPublicKeyInfo
	_, err = asn1.Unmarshal(encodedpub, &subPKI)
	if err != nil {
		return nil, err
	}

	pubhash := sha1.New()
	pubhash.Write(subPKI.SubjectPublicKey.Bytes)

	var (
		eku             []x509.ExtKeyUsage
		ku              x509.KeyUsage
		expiry          time.Duration
		crlURL, ocspURL string
	)

	// The third value returned from Usages is a list of unknown key usages.
	// This should be used when validating the profile at load, and isn't used
	// here.
	ku, eku, _ = profile.Usages()
	expiry = profile.Expiry

	if ku == 0 && len(eku) == 0 {
		err = cferr.New(cferr.PolicyError, cferr.NoKeyUsages)
		return nil, err
	}

	if expiry == 0 {
		expiry = threeMonths
	}

	now := time.Now()
	serialNumber, err := rand.Int(rand.Reader, new(big.Int).SetInt64(math.MaxInt64))
	if err != nil {
		err = cferr.Wrap(cferr.CSRError, cferr.Unknown, err)
		return nil, err
	}

	template.SerialNumber = serialNumber
	template.NotBefore = now.Add(-5 * time.Minute).UTC()
	template.NotAfter = now.Add(expiry).UTC()
	template.KeyUsage = ku
	template.ExtKeyUsage = eku
	template.BasicConstraintsValid = true
	template.IsCA = profile.CAConstraint.IsCA
	template.SubjectKeyId = pubhash.Sum(nil)

	if ocspURL != "" {
		template.OCSPServer = []string{ocspURL}
	}
	if crlURL != "" {
		template.CRLDistributionPoints = []string{crlURL}
	}

	if len(profile.IssuerURL) != 0 {
		template.IssuingCertificateURL = profile.IssuerURL
	}

	cert, err := x509.CreateCertificate(rand.Reader, template, template, pub, priv)
	if err != nil {
		return nil, err
	}

	cert = pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: cert})
	return cert, nil
}
