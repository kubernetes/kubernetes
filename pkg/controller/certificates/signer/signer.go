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
	"context"
	"crypto/x509"
	"encoding/pem"
	"fmt"
	"strings"
	"time"

	capi "k8s.io/api/certificates/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/server/dynamiccertificates"
	certificatesinformers "k8s.io/client-go/informers/certificates/v1beta1"
	clientset "k8s.io/client-go/kubernetes"
	capihelper "k8s.io/kubernetes/pkg/apis/certificates/v1beta1"
	"k8s.io/kubernetes/pkg/controller/certificates"
	"k8s.io/kubernetes/pkg/controller/certificates/authority"
)

type CSRSigningController struct {
	certificateController *certificates.CertificateController
	dynamicCertReloader   dynamiccertificates.ControllerRunner
}

func NewCSRSigningController(
	client clientset.Interface,
	csrInformer certificatesinformers.CertificateSigningRequestInformer,
	caFile, caKeyFile string,
	certTTL time.Duration,
) (*CSRSigningController, error) {
	signer, err := newSigner(caFile, caKeyFile, client, certTTL)
	if err != nil {
		return nil, err
	}

	return &CSRSigningController{
		certificateController: certificates.NewCertificateController(
			"csrsigning",
			client,
			csrInformer,
			signer.handle,
		),
		dynamicCertReloader: signer.caProvider.caLoader,
	}, nil
}

// Run the main goroutine responsible for watching and syncing jobs.
func (c *CSRSigningController) Run(workers int, stopCh <-chan struct{}) {
	go c.dynamicCertReloader.Run(workers, stopCh)

	c.certificateController.Run(workers, stopCh)
}

type signer struct {
	caProvider *caProvider

	client  clientset.Interface
	certTTL time.Duration
}

func newSigner(caFile, caKeyFile string, client clientset.Interface, certificateDuration time.Duration) (*signer, error) {
	caProvider, err := newCAProvider(caFile, caKeyFile)
	if err != nil {
		return nil, err
	}

	ret := &signer{
		caProvider: caProvider,
		client:     client,
		certTTL:    certificateDuration,
	}
	return ret, nil
}

func (s *signer) handle(csr *capi.CertificateSigningRequest) error {
	// Ignore unapproved requests
	if !certificates.IsCertificateRequestApproved(csr) {
		return nil
	}

	// Fast-path to avoid any additional processing if the CSRs signerName does
	// not have a 'kubernetes.io/' prefix.
	if !strings.HasPrefix(*csr.Spec.SignerName, "kubernetes.io/") {
		return nil
	}

	x509cr, err := capihelper.ParseCSR(csr.Spec.Request)
	if err != nil {
		return fmt.Errorf("unable to parse csr %q: %v", csr.Name, err)
	}
	if !requestValidForSignerName(x509cr, csr.Spec.Usages, *csr.Spec.SignerName) {
		// TODO: mark the CertificateRequest as being in a terminal state and
		//  communicate to the user why the request has been refused.
		return nil
	}
	cert, err := s.sign(x509cr, csr.Spec.Usages)
	if err != nil {
		return fmt.Errorf("error auto signing csr: %v", err)
	}
	csr.Status.Certificate = cert
	_, err = s.client.CertificatesV1beta1().CertificateSigningRequests().UpdateStatus(context.TODO(), csr, metav1.UpdateOptions{})
	if err != nil {
		return fmt.Errorf("error updating signature for csr: %v", err)
	}
	return nil
}

func (s *signer) sign(x509cr *x509.CertificateRequest, usages []capi.KeyUsage) ([]byte, error) {
	currCA, err := s.caProvider.currentCA()
	if err != nil {
		return nil, err
	}
	der, err := currCA.Sign(x509cr.Raw, authority.PermissiveSigningPolicy{
		TTL:    s.certTTL,
		Usages: usages,
	})
	if err != nil {
		return nil, err
	}
	return pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: der}), nil
}

func requestValidForSignerName(req *x509.CertificateRequest, usages []capi.KeyUsage, signerName string) bool {
	// Only handle CSRs with the specific known signerNames.
	switch signerName {
	case capi.KubeletServingSignerName:
		return capihelper.IsKubeletServingCSR(req, usages)
	case capi.KubeAPIServerClientKubeletSignerName:
		return capihelper.IsKubeletClientCSR(req, usages)
	case capi.KubeAPIServerClientSignerName:
		return validAPIServerClientUsages(usages)
	case capi.LegacyUnknownSignerName:
		// No restrictions are applied to the legacy-unknown signerName to
		// maintain backward compatibility in v1beta1.
		return true
	default:
		return false
	}
}

func validAPIServerClientUsages(usages []capi.KeyUsage) bool {
	hasClientAuth := false
	for _, u := range usages {
		switch u {
		// these usages are optional
		case capi.UsageDigitalSignature, capi.UsageKeyEncipherment:
		case capi.UsageClientAuth:
			hasClientAuth = true
		default:
			return false
		}
	}
	return hasClientAuth
}
