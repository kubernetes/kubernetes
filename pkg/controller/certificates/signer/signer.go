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
	"crypto"
	"encoding/pem"
	"fmt"
	"io/ioutil"
	"time"

	capi "k8s.io/api/certificates/v1beta1"
	certificatesinformers "k8s.io/client-go/informers/certificates/v1beta1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/util/cert"
	"k8s.io/client-go/util/keyutil"
	capihelper "k8s.io/kubernetes/pkg/apis/certificates/v1beta1"
	"k8s.io/kubernetes/pkg/controller/certificates"
	"k8s.io/kubernetes/pkg/controller/certificates/authority"
)

func NewCSRSigningController(
	client clientset.Interface,
	csrInformer certificatesinformers.CertificateSigningRequestInformer,
	caFile, caKeyFile string,
	certTTL time.Duration,
) (*certificates.CertificateController, error) {
	signer, err := newSigner(caFile, caKeyFile, client, certTTL)
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
	ca      *authority.CertificateAuthority
	client  clientset.Interface
	certTTL time.Duration
}

func newSigner(caFile, caKeyFile string, client clientset.Interface, certificateDuration time.Duration) (*signer, error) {
	certPEM, err := ioutil.ReadFile(caFile)
	if err != nil {
		return nil, fmt.Errorf("error reading CA cert file %q: %v", caFile, err)
	}

	certs, err := cert.ParseCertsPEM(certPEM)
	if err != nil {
		return nil, fmt.Errorf("error reading CA cert file %q: %v", caFile, err)
	}
	if len(certs) != 1 {
		return nil, fmt.Errorf("error reading CA cert file %q: expected 1 certificate, found %d", caFile, len(certs))
	}

	keyPEM, err := ioutil.ReadFile(caKeyFile)
	if err != nil {
		return nil, fmt.Errorf("error reading CA key file %q: %v", caKeyFile, err)
	}
	key, err := keyutil.ParsePrivateKeyPEM(keyPEM)
	if err != nil {
		return nil, fmt.Errorf("error reading CA key file %q: %v", caKeyFile, err)
	}
	priv, ok := key.(crypto.Signer)
	if !ok {
		return nil, fmt.Errorf("error reading CA key file %q: key did not implement crypto.Signer", caKeyFile)
	}

	return &signer{
		ca: &authority.CertificateAuthority{
			Certificate: certs[0],
			PrivateKey:  priv,
			Backdate:    5 * time.Minute,
		},
		client:  client,
		certTTL: certificateDuration,
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

	der, err := s.ca.Sign(x509cr.Raw, authority.PermissiveSigningPolicy{
		TTL:    s.certTTL,
		Usages: csr.Spec.Usages,
	})
	if err != nil {
		return nil, err
	}
	csr.Status.Certificate = pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: der})
	return csr, nil
}
