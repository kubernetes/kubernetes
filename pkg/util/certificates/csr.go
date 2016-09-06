/*
Copyright 2016 The Kubernetes Authors.

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

package certificates

import (
	"crypto/ecdsa"
	"crypto/elliptic"
	cryptorand "crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"errors"
	"fmt"
	"net"

	"k8s.io/kubernetes/pkg/apis/certificates"
)

// ParseCertificateRequestObject extracts the CSR from the API object and decodes it.
func ParseCertificateRequestObject(obj *certificates.CertificateSigningRequest) (*x509.CertificateRequest, error) {
	// extract PEM from request object
	pemBytes := obj.Spec.Request
	block, _ := pem.Decode(pemBytes)
	if block == nil || block.Type != "CERTIFICATE REQUEST" {
		return nil, errors.New("PEM block type must be CERTIFICATE REQUEST")
	}
	csr, err := x509.ParseCertificateRequest(block.Bytes)
	if err != nil {
		return nil, err
	}
	return csr, nil
}

// GeneratePrivateKey returns PEM data containing a generated ECDSA private key
func GeneratePrivateKey() ([]byte, error) {
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

// ParsePrivateKey returns a private key parsed from a PEM block in the supplied data.
// Recognizes PEM blocks for "EC PRIVATE KEY" and "RSA PRIVATE KEY"
func ParsePrivateKey(keyData []byte) (interface{}, error) {
	for {
		var privateKeyPemBlock *pem.Block
		privateKeyPemBlock, keyData = pem.Decode(keyData)
		if privateKeyPemBlock == nil {
			// we read all the PEM blocks and didn't recognize one
			return nil, fmt.Errorf("no private key PEM block found")
		}

		switch privateKeyPemBlock.Type {
		case "EC PRIVATE KEY":
			return x509.ParseECPrivateKey(privateKeyPemBlock.Bytes)
		case "RSA PRIVATE KEY":
			return x509.ParsePKCS1PrivateKey(privateKeyPemBlock.Bytes)
		}
	}
}

// NewCertificateRequest generates a PEM-encoded CSR using the supplied private key, subject, and SANs.
// privateKey must be a *ecdsa.PrivateKey or *rsa.PrivateKey.
func NewCertificateRequest(privateKey interface{}, subject *pkix.Name, dnsSANs []string, ipSANs []net.IP) (csr []byte, err error) {
	var sigType x509.SignatureAlgorithm

	switch privateKey := privateKey.(type) {
	case *ecdsa.PrivateKey:
		switch privateKey.Curve {
		case elliptic.P256():
			sigType = x509.ECDSAWithSHA256
		case elliptic.P384():
			sigType = x509.ECDSAWithSHA384
		case elliptic.P521():
			sigType = x509.ECDSAWithSHA512
		default:
			return nil, fmt.Errorf("unknown elliptic curve: %v", privateKey.Curve)
		}
	case *rsa.PrivateKey:
		keySize := privateKey.N.BitLen()
		switch {
		case keySize >= 4096:
			sigType = x509.SHA512WithRSA
		case keySize >= 3072:
			sigType = x509.SHA384WithRSA
		default:
			sigType = x509.SHA256WithRSA
		}

	default:
		return nil, fmt.Errorf("unsupported key type: %T", privateKey)
	}

	template := &x509.CertificateRequest{
		Subject:            *subject,
		SignatureAlgorithm: sigType,
		DNSNames:           dnsSANs,
		IPAddresses:        ipSANs,
	}

	csr, err = x509.CreateCertificateRequest(cryptorand.Reader, template, privateKey)
	if err != nil {
		return nil, err
	}

	csrPemBlock := &pem.Block{
		Type:  "CERTIFICATE REQUEST",
		Bytes: csr,
	}

	return pem.EncodeToMemory(csrPemBlock), nil
}
