package libtrust

import (
	"crypto/rand"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"fmt"
	"io/ioutil"
	"math/big"
	"net"
	"time"
)

type certTemplateInfo struct {
	commonName  string
	domains     []string
	ipAddresses []net.IP
	isCA        bool
	clientAuth  bool
	serverAuth  bool
}

func generateCertTemplate(info *certTemplateInfo) *x509.Certificate {
	// Generate a certificate template which is valid from the past week to
	// 10 years from now. The usage of the certificate depends on the
	// specified fields in the given certTempInfo object.
	var (
		keyUsage    x509.KeyUsage
		extKeyUsage []x509.ExtKeyUsage
	)

	if info.isCA {
		keyUsage = x509.KeyUsageCertSign
	}

	if info.clientAuth {
		extKeyUsage = append(extKeyUsage, x509.ExtKeyUsageClientAuth)
	}

	if info.serverAuth {
		extKeyUsage = append(extKeyUsage, x509.ExtKeyUsageServerAuth)
	}

	return &x509.Certificate{
		SerialNumber: big.NewInt(0),
		Subject: pkix.Name{
			CommonName: info.commonName,
		},
		NotBefore:             time.Now().Add(-time.Hour * 24 * 7),
		NotAfter:              time.Now().Add(time.Hour * 24 * 365 * 10),
		DNSNames:              info.domains,
		IPAddresses:           info.ipAddresses,
		IsCA:                  info.isCA,
		KeyUsage:              keyUsage,
		ExtKeyUsage:           extKeyUsage,
		BasicConstraintsValid: info.isCA,
	}
}

func generateCert(pub PublicKey, priv PrivateKey, subInfo, issInfo *certTemplateInfo) (cert *x509.Certificate, err error) {
	pubCertTemplate := generateCertTemplate(subInfo)
	privCertTemplate := generateCertTemplate(issInfo)

	certDER, err := x509.CreateCertificate(
		rand.Reader, pubCertTemplate, privCertTemplate,
		pub.CryptoPublicKey(), priv.CryptoPrivateKey(),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create certificate: %s", err)
	}

	cert, err = x509.ParseCertificate(certDER)
	if err != nil {
		return nil, fmt.Errorf("failed to parse certificate: %s", err)
	}

	return
}

// GenerateSelfSignedServerCert creates a self-signed certificate for the
// given key which is to be used for TLS servers with the given domains and
// IP addresses.
func GenerateSelfSignedServerCert(key PrivateKey, domains []string, ipAddresses []net.IP) (*x509.Certificate, error) {
	info := &certTemplateInfo{
		commonName:  key.KeyID(),
		domains:     domains,
		ipAddresses: ipAddresses,
		serverAuth:  true,
	}

	return generateCert(key.PublicKey(), key, info, info)
}

// GenerateSelfSignedClientCert creates a self-signed certificate for the
// given key which is to be used for TLS clients.
func GenerateSelfSignedClientCert(key PrivateKey) (*x509.Certificate, error) {
	info := &certTemplateInfo{
		commonName: key.KeyID(),
		clientAuth: true,
	}

	return generateCert(key.PublicKey(), key, info, info)
}

// GenerateCACert creates a certificate which can be used as a trusted
// certificate authority.
func GenerateCACert(signer PrivateKey, trustedKey PublicKey) (*x509.Certificate, error) {
	subjectInfo := &certTemplateInfo{
		commonName: trustedKey.KeyID(),
		isCA:       true,
	}
	issuerInfo := &certTemplateInfo{
		commonName: signer.KeyID(),
	}

	return generateCert(trustedKey, signer, subjectInfo, issuerInfo)
}

// GenerateCACertPool creates a certificate authority pool to be used for a
// TLS configuration. Any self-signed certificates issued by the specified
// trusted keys will be verified during a TLS handshake
func GenerateCACertPool(signer PrivateKey, trustedKeys []PublicKey) (*x509.CertPool, error) {
	certPool := x509.NewCertPool()

	for _, trustedKey := range trustedKeys {
		cert, err := GenerateCACert(signer, trustedKey)
		if err != nil {
			return nil, fmt.Errorf("failed to generate CA certificate: %s", err)
		}

		certPool.AddCert(cert)
	}

	return certPool, nil
}

// LoadCertificateBundle loads certificates from the given file.  The file should be pem encoded
// containing one or more certificates.  The expected pem type is "CERTIFICATE".
func LoadCertificateBundle(filename string) ([]*x509.Certificate, error) {
	b, err := ioutil.ReadFile(filename)
	if err != nil {
		return nil, err
	}
	certificates := []*x509.Certificate{}
	var block *pem.Block
	block, b = pem.Decode(b)
	for ; block != nil; block, b = pem.Decode(b) {
		if block.Type == "CERTIFICATE" {
			cert, err := x509.ParseCertificate(block.Bytes)
			if err != nil {
				return nil, err
			}
			certificates = append(certificates, cert)
		} else {
			return nil, fmt.Errorf("invalid pem block type: %s", block.Type)
		}
	}

	return certificates, nil
}

// LoadCertificatePool loads a CA pool from the given file.  The file should be pem encoded
// containing one or more certificates. The expected pem type is "CERTIFICATE".
func LoadCertificatePool(filename string) (*x509.CertPool, error) {
	certs, err := LoadCertificateBundle(filename)
	if err != nil {
		return nil, err
	}
	pool := x509.NewCertPool()
	for _, cert := range certs {
		pool.AddCert(cert)
	}
	return pool, nil
}
