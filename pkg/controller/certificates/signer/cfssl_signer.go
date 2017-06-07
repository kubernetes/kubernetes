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

// Package signer implements a CA signer that uses keys stored on local disk.
package signer

import (
	"crypto"
	"crypto/x509"
	"fmt"
	"io/ioutil"
	"os"
	"time"

	capi "k8s.io/kubernetes/pkg/apis/certificates/v1beta1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	certificatesinformers "k8s.io/kubernetes/pkg/client/informers/informers_generated/externalversions/certificates/v1beta1"
	"k8s.io/kubernetes/pkg/controller/certificates"

	"github.com/cloudflare/cfssl/config"
	"github.com/cloudflare/cfssl/helpers"
	"github.com/cloudflare/cfssl/signer"
	"github.com/cloudflare/cfssl/signer/local"
)

func NewCSRSigningController(
	client clientset.Interface,
	csrInformer certificatesinformers.CertificateSigningRequestInformer,
	caFile, caKeyFile string,
	certificateDuration time.Duration,
) (*certificates.CertificateController, error) {
	signer, err := newCFSSLSigner(caFile, caKeyFile, client, certificateDuration)
	if err != nil {
		return nil, err
	}
	return certificates.NewCertificateController(
		client,
		csrInformer,
		signer.handle,
	)
}

type cfsslSigner struct {
	ca                  *x509.Certificate
	priv                crypto.Signer
	sigAlgo             x509.SignatureAlgorithm
	client              clientset.Interface
	certificateDuration time.Duration
}

func newCFSSLSigner(caFile, caKeyFile string, client clientset.Interface, certificateDuration time.Duration) (*cfsslSigner, error) {
	ca, err := ioutil.ReadFile(caFile)
	if err != nil {
		return nil, fmt.Errorf("error reading CA cert file %q: %v", caFile, err)
	}
	cakey, err := ioutil.ReadFile(caKeyFile)
	if err != nil {
		return nil, fmt.Errorf("error reading CA key file %q: %v", caKeyFile, err)
	}

	parsedCa, err := helpers.ParseCertificatePEM(ca)
	if err != nil {
		return nil, fmt.Errorf("error parsing CA cert file %q: %v", caFile, err)
	}

	strPassword := os.Getenv("CFSSL_CA_PK_PASSWORD")
	password := []byte(strPassword)
	if strPassword == "" {
		password = nil
	}

	priv, err := helpers.ParsePrivateKeyPEMWithPassword(cakey, password)
	if err != nil {
		return nil, fmt.Errorf("Malformed private key %v", err)
	}
	return &cfsslSigner{
		priv:                priv,
		ca:                  parsedCa,
		sigAlgo:             signer.DefaultSigAlgo(priv),
		client:              client,
		certificateDuration: certificateDuration,
	}, nil
}

func (s *cfsslSigner) handle(csr *capi.CertificateSigningRequest) error {
	if !certificates.IsCertificateRequestApproved(csr) {
		return nil
	}
	csr, err := s.sign(csr)
	if err != nil {
		return fmt.Errorf("error auto signing csr: %v", err)
	}
	_, err = s.client.Certificates().CertificateSigningRequests().UpdateStatus(csr)
	if err != nil {
		return fmt.Errorf("error updating signature for csr: %v", err)
	}
	return nil
}

func (s *cfsslSigner) sign(csr *capi.CertificateSigningRequest) (*capi.CertificateSigningRequest, error) {
	var usages []string
	for _, usage := range csr.Spec.Usages {
		usages = append(usages, string(usage))
	}
	policy := &config.Signing{
		Default: &config.SigningProfile{
			Usage:        usages,
			Expiry:       s.certificateDuration,
			ExpiryString: s.certificateDuration.String(),
		},
	}
	cfs, err := local.NewSigner(s.priv, s.ca, s.sigAlgo, policy)
	if err != nil {
		return nil, err
	}

	csr.Status.Certificate, err = cfs.Sign(signer.SignRequest{
		Request: string(csr.Spec.Request),
	})
	if err != nil {
		return nil, err
	}

	return csr, nil
}
