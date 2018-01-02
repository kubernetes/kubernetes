package testutil

import (
	"crypto"
	"crypto/rand"
	"crypto/x509"
	"crypto/x509/pkix"
	"math/big"
	"time"
)

// GenerateTrustCA generates a new certificate authority for testing.
func GenerateTrustCA(pub crypto.PublicKey, priv crypto.PrivateKey) (*x509.Certificate, error) {
	cert := &x509.Certificate{
		SerialNumber: big.NewInt(0),
		Subject: pkix.Name{
			CommonName: "CA Root",
		},
		NotBefore:             time.Now().Add(-time.Second),
		NotAfter:              time.Now().Add(time.Hour),
		IsCA:                  true,
		KeyUsage:              x509.KeyUsageCertSign | x509.KeyUsageCRLSign,
		BasicConstraintsValid: true,
	}

	certDER, err := x509.CreateCertificate(rand.Reader, cert, cert, pub, priv)
	if err != nil {
		return nil, err
	}

	cert, err = x509.ParseCertificate(certDER)
	if err != nil {
		return nil, err
	}

	return cert, nil
}

// GenerateIntermediate generates an intermediate certificate for testing using
// the parent certificate (likely a CA) and the provided keys.
func GenerateIntermediate(key crypto.PublicKey, parentKey crypto.PrivateKey, parent *x509.Certificate) (*x509.Certificate, error) {
	cert := &x509.Certificate{
		SerialNumber: big.NewInt(0),
		Subject: pkix.Name{
			CommonName: "Intermediate",
		},
		NotBefore:             time.Now().Add(-time.Second),
		NotAfter:              time.Now().Add(time.Hour),
		IsCA:                  true,
		KeyUsage:              x509.KeyUsageCertSign | x509.KeyUsageCRLSign,
		BasicConstraintsValid: true,
	}

	certDER, err := x509.CreateCertificate(rand.Reader, cert, parent, key, parentKey)
	if err != nil {
		return nil, err
	}

	cert, err = x509.ParseCertificate(certDER)
	if err != nil {
		return nil, err
	}

	return cert, nil
}

// GenerateTrustCert generates a new trust certificate for testing.  Unlike the
// intermediate certificates, this certificate should  be used for signature
// only, not creating certificates.
func GenerateTrustCert(key crypto.PublicKey, parentKey crypto.PrivateKey, parent *x509.Certificate) (*x509.Certificate, error) {
	cert := &x509.Certificate{
		SerialNumber: big.NewInt(0),
		Subject: pkix.Name{
			CommonName: "Trust Cert",
		},
		NotBefore:             time.Now().Add(-time.Second),
		NotAfter:              time.Now().Add(time.Hour),
		IsCA:                  true,
		KeyUsage:              x509.KeyUsageDigitalSignature,
		BasicConstraintsValid: true,
	}

	certDER, err := x509.CreateCertificate(rand.Reader, cert, parent, key, parentKey)
	if err != nil {
		return nil, err
	}

	cert, err = x509.ParseCertificate(certDER)
	if err != nil {
		return nil, err
	}

	return cert, nil
}
