/*
Copyright 2014 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package cert

import (
	"crypto/ecdsa"
	"crypto/elliptic"
	cryptorand "crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"fmt"
	"math"
	"math/big"
	"net"
	"time"

	"github.com/cloudflare/cfssl/config"
	"github.com/cloudflare/cfssl/csr"
	"github.com/cloudflare/cfssl/helpers"
	"github.com/cloudflare/cfssl/initca"
	"github.com/cloudflare/cfssl/signer"
	"github.com/cloudflare/cfssl/signer/local"

	"github.com/golang/glog"
)

const (
	rsaKeySize   = 2048
	duration365d = time.Hour * 24 * 365
)

// Config containes the basic fields required for creating a certificate
type Config struct {
	CommonName   string
	Organization []string
	AltNames     AltNames
}

// AltNames contains the domain names and IP addresses that will be added
// to the API Server's x509 certificate SubAltNames field. The values will
// be passed directly to the x509.Certificate object.
type AltNames struct {
	DNSNames []string
	IPs      []net.IP
}

// NewPrivateKey creates an RSA private key
func NewPrivateKey() (*rsa.PrivateKey, error) {
	return rsa.GenerateKey(cryptorand.Reader, rsaKeySize)
}

// NewSelfSignedCACert creates a CA certificate
func NewSelfSignedCACert(cfg Config, key *rsa.PrivateKey) (*x509.Certificate, error) {
	now := time.Now()
	tmpl := x509.Certificate{
		SerialNumber: new(big.Int).SetInt64(0),
		Subject: pkix.Name{
			CommonName:   cfg.CommonName,
			Organization: cfg.Organization,
		},
		NotBefore:             now.UTC(),
		NotAfter:              now.Add(duration365d * 10).UTC(),
		KeyUsage:              x509.KeyUsageKeyEncipherment | x509.KeyUsageDigitalSignature | x509.KeyUsageCertSign,
		BasicConstraintsValid: true,
		IsCA: true,
	}

	certDERBytes, err := x509.CreateCertificate(cryptorand.Reader, &tmpl, &tmpl, key.Public(), key)
	if err != nil {
		return nil, err
	}
	return x509.ParseCertificate(certDERBytes)
}

// NewSignedCert creates a signed certificate using the given CA certificate and key
func NewSignedCert(cfg Config, key *rsa.PrivateKey, caCert *x509.Certificate, caKey *rsa.PrivateKey) (*x509.Certificate, error) {
	serial, err := cryptorand.Int(cryptorand.Reader, new(big.Int).SetInt64(math.MaxInt64))
	if err != nil {
		return nil, err
	}

	certTmpl := x509.Certificate{
		Subject: pkix.Name{
			CommonName:   cfg.CommonName,
			Organization: caCert.Subject.Organization,
		},
		DNSNames:     cfg.AltNames.DNSNames,
		IPAddresses:  cfg.AltNames.IPs,
		SerialNumber: serial,
		NotBefore:    caCert.NotBefore,
		NotAfter:     time.Now().Add(duration365d).UTC(),
		KeyUsage:     x509.KeyUsageKeyEncipherment | x509.KeyUsageDigitalSignature,
		ExtKeyUsage:  []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth, x509.ExtKeyUsageClientAuth},
	}
	certDERBytes, err := x509.CreateCertificate(cryptorand.Reader, &certTmpl, caCert, key.Public(), caKey)
	if err != nil {
		return nil, err
	}
	return x509.ParseCertificate(certDERBytes)
}

// MakeEllipticPrivateKeyPEM creates an ECDSA private key
func MakeEllipticPrivateKeyPEM() ([]byte, error) {
	privateKey, err := ecdsa.GenerateKey(elliptic.P256(), cryptorand.Reader)
	if err != nil {
		return nil, err
	}

	derBytes, err := x509.MarshalECPrivateKey(privateKey)
	if err != nil {
		return nil, err
	}

	privateKeyPemBlock := &pem.Block{
		Type:  "EC PRIVATE KEY",
		Bytes: derBytes,
	}
	return pem.EncodeToMemory(privateKeyPemBlock), nil
}

// GenerateSelfSignedCertKey creates a self-signed CA, a server key and certificate signed
// with the CA. Host may be an IP or a DNS name. You may also specify additional subject
// alt names (either ip or dns names) for the certificate.
func GenerateSelfSignedCertKey(host string, alternateIPs []net.IP, alternateDNS []string) (caCertPem []byte, caKeyPem []byte, certPem []byte, keyPem []byte, err error) {
	// create root CA
	glog.Infof("Creating root certificate authority")
	rootCAReq := csr.CertificateRequest{
		CN: fmt.Sprintf("%s@ca-%d", host, time.Now().Unix()),
		KeyRequest: &csr.BasicKeyRequest{
			A: "rsa",
			S: 2048,
		},
		CA: &csr.CAConfig{
			Expiry: fmt.Sprintf("%dh", 24*365*1),
		},
	}
	caCertPem, _, caKeyPem, err = initca.New(&rootCAReq)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("error creating root certificate authority: %v", err)
	}

	// create key and csr
	glog.Infof("Creating signing request for apiserver key")
	hosts := []string{host}
	for _, ip := range alternateIPs {
		hosts = append(hosts, ip.String())
	}
	hosts = append(hosts, alternateDNS...)
	req := csr.CertificateRequest{
		CN: fmt.Sprintf("%s@%d", host, time.Now().Unix()),
		KeyRequest: &csr.BasicKeyRequest{
			A: "rsa",
			S: 2048,
		},
		CA: &csr.CAConfig{
			Expiry: fmt.Sprintf("%dh", 24*365*1),
		},
		Hosts: hosts,
	}

	glog.Infof("Creating apiserver key and certificate signing request")
	gen := csr.Generator{
		Validator: func(req *csr.CertificateRequest) error {
			return nil
		},
	}
	keyCSR, keyPem, err := gen.ProcessRequest(&req)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("error creating key and certificate signing request: %v", err)
	}

	// sign key with root CA
	caKey, err := helpers.ParsePrivateKeyPEM(caKeyPem)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("failed to parse private certificate authority key: %v", err)
	}
	caCert, err := helpers.ParseCertificatePEM(caCertPem)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("failed to parse root certificate authority certificate: %v", err)
	}
	glog.Infof("Signing apiserver certificate with root certificate authority")
	policy := config.Signing{
		Profiles: map[string]*config.SigningProfile{},
		Default:  config.DefaultConfig(),
	}
	policy.Default.ExpiryString = fmt.Sprintf("%dh", 24*365*1)
	s, err := local.NewSigner(caKey, caCert, signer.DefaultSigAlgo(caKey), &policy)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("error creating signer: %v", err)
	}
	certPem, err = s.Sign(signer.SignRequest{
		Request: string(keyCSR),
	})
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("error signing apiserver certificate: %v", err)
	}

	return caCertPem, caKeyPem, certPem, keyPem, nil
}

// FormatBytesCert receives byte array certificate and formats in human-readable format
func FormatBytesCert(cert []byte) (string, error) {
	block, _ := pem.Decode(cert)
	c, err := x509.ParseCertificate(block.Bytes)
	if err != nil {
		return "", fmt.Errorf("failed to parse certificate [%v]", err)
	}
	return FormatCert(c), nil
}

// FormatCert receives certificate and formats in human-readable format
func FormatCert(c *x509.Certificate) string {
	var ips []string
	for _, ip := range c.IPAddresses {
		ips = append(ips, ip.String())
	}
	altNames := append(ips, c.DNSNames...)
	res := fmt.Sprintf(
		"Issuer: CN=%s | Subject: CN=%s | CA: %t\n",
		c.Issuer.CommonName, c.Subject.CommonName, c.IsCA,
	)
	res += fmt.Sprintf("Not before: %s Not After: %s", c.NotBefore, c.NotAfter)
	if len(altNames) > 0 {
		res += fmt.Sprintf("\nAlternate Names: %v", altNames)
	}
	return res
}
