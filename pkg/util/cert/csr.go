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

package cert

import (
	cryptorand "crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"errors"
	"net"

	"k8s.io/kubernetes/pkg/apis/certificates"
	"k8s.io/kubernetes/pkg/apis/certificates/v1alpha1"
)

// ParseCSR extracts the CSR from the API object and decodes it.
func ParseCSR(obj *certificates.CertificateSigningRequest) (*x509.CertificateRequest, error) {
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

// ParseCSRV1alpha1 extracts the CSR from the API object and decodes it.
func ParseCSRV1alpha1(obj *v1alpha1.CertificateSigningRequest) (*x509.CertificateRequest, error) {
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

// MakeCSR generates a PEM-encoded CSR using the supplied private key, subject, and SANs.
// All key types that are implemented via crypto.Signer are supported (This includes *rsa.PrivateKey and *ecdsa.PrivateKey.)
func MakeCSR(privateKey interface{}, subject *pkix.Name, dnsSANs []string, ipSANs []net.IP) (csr []byte, err error) {
	// Customize the signature for RSA keys, depending on the key size
	var sigType x509.SignatureAlgorithm
	if privateKey, ok := privateKey.(*rsa.PrivateKey); ok {
		keySize := privateKey.N.BitLen()
		switch {
		case keySize >= 4096:
			sigType = x509.SHA512WithRSA
		case keySize >= 3072:
			sigType = x509.SHA384WithRSA
		default:
			sigType = x509.SHA256WithRSA
		}
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
