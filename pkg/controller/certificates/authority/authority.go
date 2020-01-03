/*
Copyright 2019 The Kubernetes Authors.

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

package authority

import (
	"crypto"
	"crypto/rand"
	"crypto/x509"
	"fmt"
	"math/big"
	"time"
)

var serialNumberLimit = new(big.Int).Lsh(big.NewInt(1), 128)

// CertificateAuthority implements a certificate authority that supports policy
// based signing. It's used by the signing controller.
type CertificateAuthority struct {
	// RawCert is an optional field to determine if signing cert/key pairs have changed
	RawCert []byte
	// RawKey is an optional field to determine if signing cert/key pairs have changed
	RawKey []byte

	Certificate *x509.Certificate
	PrivateKey  crypto.Signer
	Backdate    time.Duration
	Now         func() time.Time
}

// Sign signs a certificate request, applying a SigningPolicy and returns a DER
// encoded x509 certificate.
func (ca *CertificateAuthority) Sign(crDER []byte, policy SigningPolicy) ([]byte, error) {
	now := time.Now()
	if ca.Now != nil {
		now = ca.Now()
	}

	nbf := now.Add(-ca.Backdate)
	if !nbf.Before(ca.Certificate.NotAfter) {
		return nil, fmt.Errorf("the signer has expired: NotAfter=%v", ca.Certificate.NotAfter)
	}

	cr, err := x509.ParseCertificateRequest(crDER)
	if err != nil {
		return nil, fmt.Errorf("unable to parse certificate request: %v", err)
	}
	if err := cr.CheckSignature(); err != nil {
		return nil, fmt.Errorf("unable to verify certificate request signature: %v", err)
	}

	serialNumber, err := rand.Int(rand.Reader, serialNumberLimit)
	if err != nil {
		return nil, fmt.Errorf("unable to generate a serial number for %s: %v", cr.Subject.CommonName, err)
	}

	tmpl := &x509.Certificate{
		SerialNumber:       serialNumber,
		Subject:            cr.Subject,
		DNSNames:           cr.DNSNames,
		IPAddresses:        cr.IPAddresses,
		EmailAddresses:     cr.EmailAddresses,
		URIs:               cr.URIs,
		PublicKeyAlgorithm: cr.PublicKeyAlgorithm,
		PublicKey:          cr.PublicKey,
		Extensions:         cr.Extensions,
		ExtraExtensions:    cr.ExtraExtensions,
		NotBefore:          nbf,
	}
	if err := policy.apply(tmpl); err != nil {
		return nil, err
	}

	if !tmpl.NotAfter.Before(ca.Certificate.NotAfter) {
		tmpl.NotAfter = ca.Certificate.NotAfter
	}
	if !now.Before(ca.Certificate.NotAfter) {
		return nil, fmt.Errorf("refusing to sign a certificate that expired in the past")
	}

	der, err := x509.CreateCertificate(rand.Reader, tmpl, ca.Certificate, cr.PublicKey, ca.PrivateKey)
	if err != nil {
		return nil, fmt.Errorf("failed to sign certificate: %v", err)
	}
	return der, nil
}
