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

// Package signer implements a CA signer that uses keys stored on local disk.
package signer

import (
	"crypto/rand"
	"crypto/x509"
	"encoding/pem"
	"fmt"
	"io/ioutil"
	"math/big"
	"time"

	capi "k8s.io/api/certificates/v1beta1"
	certificatesinformers "k8s.io/client-go/informers/certificates/v1beta1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/util/cert"
	"k8s.io/client-go/util/keyutil"
	capihelper "k8s.io/kubernetes/pkg/apis/certificates/v1beta1"
	"k8s.io/kubernetes/pkg/controller/certificates"
)

func NewCSRSigningController(
	client clientset.Interface,
	csrInformer certificatesinformers.CertificateSigningRequestInformer,
	caFile, caKeyFile string,
	certificateDuration time.Duration,
) (*certificates.CertificateController, error) {
	signer, err := newSigner(caFile, caKeyFile, client, certificateDuration)
	if err != nil {
		return nil, err
	}
	return certificates.NewCertificateController(
		"csrsigning",
		client,
		csrInformer,
		signer.handle,
	), nil
}

type signer struct {
	ca      *x509.Certificate
	priv    interface{}
	client  clientset.Interface
	certTTL time.Duration

	// now is mocked for testing.
	now func() time.Time
}

func newSigner(caFile, caKeyFile string, client clientset.Interface, certificateDuration time.Duration) (*signer, error) {
	ca, err := ioutil.ReadFile(caFile)
	if err != nil {
		return nil, fmt.Errorf("error reading CA cert file %q: %v", caFile, err)
	}
	cakey, err := ioutil.ReadFile(caKeyFile)
	if err != nil {
		return nil, fmt.Errorf("error reading CA key file %q: %v", caKeyFile, err)
	}

	certs, err := cert.ParseCertsPEM(ca)
	if err != nil {
		return nil, fmt.Errorf("error parsing CA cert file %q: %v", caFile, err)
	}
	if len(certs) != 1 {
		return nil, fmt.Errorf("error parsing CA cert file %q: expected one certificate block", caFile)
	}

	priv, err := keyutil.ParsePrivateKeyPEM(cakey)
	if err != nil {
		return nil, fmt.Errorf("malformed private key %v", err)
	}
	return &signer{
		priv:    priv,
		ca:      certs[0],
		client:  client,
		certTTL: certificateDuration,
		now:     time.Now,
	}, nil
}

func (s *signer) handle(csr *capi.CertificateSigningRequest) error {
	if !certificates.IsCertificateRequestApproved(csr) {
		return nil
	}
	csr, err := s.sign(csr)
	if err != nil {
		return fmt.Errorf("error auto signing csr: %v", err)
	}
	_, err = s.client.CertificatesV1beta1().CertificateSigningRequests().UpdateStatus(csr)
	if err != nil {
		return fmt.Errorf("error updating signature for csr: %v", err)
	}
	return nil
}

func (s *signer) sign(csr *capi.CertificateSigningRequest) (*capi.CertificateSigningRequest, error) {
	x509cr, err := capihelper.ParseCSR(csr)
	if err != nil {
		return nil, fmt.Errorf("unable to parse csr %q: %v", csr.Name, err)
	}

	usage, extUsages, err := keyUsagesFromStrings(csr.Spec.Usages)
	if err != nil {
		return nil, err
	}

	now := s.now()
	expiry := now.Add(s.certTTL)
	if s.ca.NotAfter.Before(expiry) {
		expiry = s.ca.NotAfter
	}
	if expiry.Before(now) {
		return nil, fmt.Errorf("the signer has expired: NotAfter=%v", s.ca.NotAfter)
	}

	serialNumberLimit := new(big.Int).Lsh(big.NewInt(1), 128)
	serialNumber, err := rand.Int(rand.Reader, serialNumberLimit)
	if err != nil {
		return nil, fmt.Errorf("unable to generate a serial number for %s: %v", x509cr.Subject.CommonName, err)
	}

	template := &x509.Certificate{
		SerialNumber:          serialNumber,
		Subject:               x509cr.Subject,
		DNSNames:              x509cr.DNSNames,
		IPAddresses:           x509cr.IPAddresses,
		EmailAddresses:        x509cr.EmailAddresses,
		URIs:                  x509cr.URIs,
		PublicKeyAlgorithm:    x509cr.PublicKeyAlgorithm,
		PublicKey:             x509cr.PublicKey,
		NotBefore:             now,
		NotAfter:              expiry,
		KeyUsage:              usage,
		ExtKeyUsage:           extUsages,
		BasicConstraintsValid: true,
		IsCA:                  false,
	}
	der, err := x509.CreateCertificate(rand.Reader, template, s.ca, x509cr.PublicKey, s.priv)
	if err != nil {
		return nil, err
	}
	csr.Status.Certificate = pem.EncodeToMemory(&pem.Block{Type: cert.CertificateBlockType, Bytes: der})
	_ = der

	return csr, nil
}

var keyUsageDict = map[capi.KeyUsage]x509.KeyUsage{
	capi.UsageSigning:           x509.KeyUsageDigitalSignature,
	capi.UsageDigitalSignature:  x509.KeyUsageDigitalSignature,
	capi.UsageContentCommitment: x509.KeyUsageContentCommitment,
	capi.UsageKeyEncipherment:   x509.KeyUsageKeyEncipherment,
	capi.UsageKeyAgreement:      x509.KeyUsageKeyAgreement,
	capi.UsageDataEncipherment:  x509.KeyUsageDataEncipherment,
	capi.UsageCertSign:          x509.KeyUsageCertSign,
	capi.UsageCRLSign:           x509.KeyUsageCRLSign,
	capi.UsageEncipherOnly:      x509.KeyUsageEncipherOnly,
	capi.UsageDecipherOnly:      x509.KeyUsageDecipherOnly,
}

var extKeyUsageDict = map[capi.KeyUsage]x509.ExtKeyUsage{
	capi.UsageAny:             x509.ExtKeyUsageAny,
	capi.UsageServerAuth:      x509.ExtKeyUsageServerAuth,
	capi.UsageClientAuth:      x509.ExtKeyUsageClientAuth,
	capi.UsageCodeSigning:     x509.ExtKeyUsageCodeSigning,
	capi.UsageEmailProtection: x509.ExtKeyUsageEmailProtection,
	capi.UsageSMIME:           x509.ExtKeyUsageEmailProtection,
	capi.UsageIPsecEndSystem:  x509.ExtKeyUsageIPSECEndSystem,
	capi.UsageIPsecTunnel:     x509.ExtKeyUsageIPSECTunnel,
	capi.UsageIPsecUser:       x509.ExtKeyUsageIPSECUser,
	capi.UsageTimestamping:    x509.ExtKeyUsageTimeStamping,
	capi.UsageOCSPSigning:     x509.ExtKeyUsageOCSPSigning,
	capi.UsageMicrosoftSGC:    x509.ExtKeyUsageMicrosoftServerGatedCrypto,
	capi.UsageNetscapeSGC:     x509.ExtKeyUsageNetscapeServerGatedCrypto,
}

// keyUsagesFromStrings will translate a slice of usage strings from the
// certificates API ("pkg/apis/certificates".KeyUsage) to x509.KeyUsage and
// x509.ExtKeyUsage types.
func keyUsagesFromStrings(usages []capi.KeyUsage) (x509.KeyUsage, []x509.ExtKeyUsage, error) {
	var keyUsage x509.KeyUsage
	var extKeyUsage []x509.ExtKeyUsage
	var unrecognized []capi.KeyUsage
	for _, usage := range usages {
		if val, ok := keyUsageDict[usage]; ok {
			keyUsage |= val
		} else if val, ok := extKeyUsageDict[usage]; ok {
			extKeyUsage = append(extKeyUsage, val)
		} else {
			unrecognized = append(unrecognized, usage)
		}
	}

	if len(unrecognized) > 0 {
		return 0, nil, fmt.Errorf("unrecognized usage values: %q", unrecognized)
	}

	return keyUsage, extKeyUsage, nil
}
