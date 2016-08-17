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

// NewCertificateRequest generates a PEM-encoded CSR using the supplied private
// key data, subject, and SANs. If the private key data is empty, it generates a
// new ECDSA P256 key to use and returns it together with the CSR data.
func NewCertificateRequest(keyData []byte, subject *pkix.Name, dnsSANs []string, ipSANs []net.IP) (csr []byte, keyData []byte, err error) {
	var privateKey interface{}
	var privateKeyPemBlock *pem.Block

	if len(keyData) == 0 {
		privateKey, err = ecdsa.GenerateKey(elliptic.P256(), cryptorand.Reader)
		if err != nil {
			return nil, nil, err
		}

		ecdsaKey := privateKey.(*ecdsa.PrivateKey)
		derBytes, err := x509.MarshalECPrivateKey(ecdsaKey)
		if err != nil {
			return nil, nil, err
		}

		privateKeyPemBlock = &pem.Block{
			Type:  "EC PRIVATE KEY",
			Bytes: derBytes,
		}
	} else {
		privateKeyPemBlock, err = pem.Decode(keyData)
		if err != nil {
			return nil, nil, err
		}
	}

	var sigType x509.SignatureAlgorithm

	switch privateKeyPemBlock.Type {
	case "EC PRIVATE KEY":
		privateKey, err = x509.ParseECPrivateKey(privateKeyPemBlock.Bytes)
		if err != nil {
			return nil, nil, err
		}
		ecdsaKey := privateKey.(*ecdsa.PrivateKey)
		switch ecdsaKey.Curve.Params().BitSize {
		case 521:
			sigType = x509.ECDSAWithSHA512
		case 384:
			sigType = x509.ECDSAWithSHA384
		default:
			sigType = x509.ECDSAWithSHA256
		}
	case "RSA PRIVATE KEY":
		privateKey, err = x509.ParsePKCS1PrivateKey(privateKeyPemBlock.Bytes)
		if err != nil {
			return nil, nil, err
		}
		rsaKey := privateKey.(*rsa.PrivateKey)
		keySize := rsaKey.N.BitLen()
		switch {
		case keySize >= 4096:
			sigType = x509.SHA512WithRSA
		case keySize >= 3072:
			sigType = x509.SHA384WithRSA
		default:
			sigType = x509.SHA256WithRSA
		}
	default:
		return nil, nil, fmt.Errorf("unsupported key type: %s", privateKeyPemBlock.Type)
	}

	template := &x509.CertificateRequest{
		Subject:            *subject,
		SignatureAlgorithm: sigType,
		DNSNames:           dnsSANs,
		IPAddresses:        ipSANs,
	}

	csr, err := x509.CreateCertificateRequest(cryptorand.Reader, template, privateKey)
	if err != nil {
		return nil, nil, err
	}

	csrPemBlock := &pem.Block{
		Type:  "CERTIFICATE REQUEST",
		Bytes: csr,
	}

	return pem.EncodeToMemory(csrPemBlock), pem.EncodeToMemory(privateKeyPemBlock), nil
}
