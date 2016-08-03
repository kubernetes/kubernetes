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
	"io/ioutil"
	"net"
	"os"

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
// key file, subject, and SANs. If the private key file does not exist, it generates a
// new ECDSA P256 key to use and writes it to the keyFile path.
func NewCertificateRequest(keyFile string, subject *pkix.Name, dnsSANs []string, ipSANs []net.IP) ([]byte, error) {
	var privateKey interface{}

	if _, err := os.Stat(keyFile); os.IsNotExist(err) {
		privateKey, err = ecdsa.GenerateKey(elliptic.P256(), cryptorand.Reader)
		if err != nil {
			return nil, err
		}

		ecdsaKey := privateKey.(*ecdsa.PrivateKey)
		derBytes, err := x509.MarshalECPrivateKey(ecdsaKey)
		if err != nil {
			return nil, err
		}

		pemBlock := &pem.Block{
			Type:  "EC PRIVATE KEY",
			Bytes: derBytes,
		}

		err = ioutil.WriteFile(keyFile, pem.EncodeToMemory(pemBlock), os.FileMode(0600))
		if err != nil {
			return nil, err
		}
	}

	keyBytes, err := ioutil.ReadFile(keyFile)
	if err != nil {
		return nil, err
	}

	var block *pem.Block
	var sigType x509.SignatureAlgorithm

	block, _ = pem.Decode(keyBytes)

	switch block.Type {
	case "EC PRIVATE KEY":
		privateKey, err = x509.ParseECPrivateKey(block.Bytes)
		if err != nil {
			return nil, err
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
		privateKey, err = x509.ParsePKCS1PrivateKey(block.Bytes)
		if err != nil {
			return nil, err
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
		return nil, fmt.Errorf("unsupported key type: %s", block.Type)
	}

	template := &x509.CertificateRequest{
		Subject:            *subject,
		SignatureAlgorithm: sigType,
		DNSNames:           dnsSANs,
		IPAddresses:        ipSANs,
	}

	csr, err := x509.CreateCertificateRequest(cryptorand.Reader, template, privateKey)
	if err != nil {
		return nil, err
	}

	pemBlock := &pem.Block{
		Type:  "CERTIFICATE REQUEST",
		Bytes: csr,
	}

	return pem.EncodeToMemory(pemBlock), nil
}
