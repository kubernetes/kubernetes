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

func NewKubeletServingCSRSigningController(
	client clientset.Interface,
	csrInformer certificatesinformers.CertificateSigningRequestInformer,
	caFile, caKeyFile string,
	certTTL time.Duration,
) (*CSRSigningController, error) {
	return NewCSRSigningController("csrsigning-kubelet-serving", capi.KubeletServingSignerName, client, csrInformer, caFile, caKeyFile, certTTL)
}

func NewKubeletClientCSRSigningController(
	client clientset.Interface,
	csrInformer certificatesinformers.CertificateSigningRequestInformer,
	caFile, caKeyFile string,
	certTTL time.Duration,
) (*CSRSigningController, error) {
	return NewCSRSigningController("csrsigning-kubelet-client", capi.KubeAPIServerClientKubeletSignerName, client, csrInformer, caFile, caKeyFile, certTTL)
}

func NewKubeAPIServerClientCSRSigningController(
	client clientset.Interface,
	csrInformer certificatesinformers.CertificateSigningRequestInformer,
	caFile, caKeyFile string,
	certTTL time.Duration,
) (*CSRSigningController, error) {
	return NewCSRSigningController("csrsigning-kube-apiserver-client", capi.KubeAPIServerClientSignerName, client, csrInformer, caFile, caKeyFile, certTTL)
}

func NewLegacyUnknownCSRSigningController(
	client clientset.Interface,
	csrInformer certificatesinformers.CertificateSigningRequestInformer,
	caFile, caKeyFile string,
	certTTL time.Duration,
) (*CSRSigningController, error) {
	return NewCSRSigningController("csrsigning-legacy-unknown", capi.LegacyUnknownSignerName, client, csrInformer, caFile, caKeyFile, certTTL)
}

func NewCSRSigningController(
	controllerName string,
	signerName string,
	client clientset.Interface,
	csrInformer certificatesinformers.CertificateSigningRequestInformer,
	caFile, caKeyFile string,
	certTTL time.Duration,
) (*CSRSigningController, error) {
	signer, err := newSigner(signerName, caFile, caKeyFile, client, certTTL)
	if err != nil {
		return nil, err
	}

	return &CSRSigningController{
		certificateController: certificates.NewCertificateController(
			controllerName,
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

type isRequestForSignerFunc func(req *x509.CertificateRequest, usages []capi.KeyUsage, signerName string) bool

type signer struct {
	caProvider *caProvider

	client  clientset.Interface
	certTTL time.Duration

	signerName           string
	isRequestForSignerFn isRequestForSignerFunc
}

func newSigner(signerName, caFile, caKeyFile string, client clientset.Interface, certificateDuration time.Duration) (*signer, error) {
	isRequestForSignerFn, err := getCSRVerificationFuncForSignerName(signerName)
	if err != nil {
		return nil, err
	}
	caProvider, err := newCAProvider(caFile, caKeyFile)
	if err != nil {
		return nil, err
	}

	ret := &signer{
		caProvider:           caProvider,
		client:               client,
		certTTL:              certificateDuration,
		signerName:           signerName,
		isRequestForSignerFn: isRequestForSignerFn,
	}
	return ret, nil
}

func (s *signer) handle(csr *capi.CertificateSigningRequest) error {
	// Ignore unapproved requests
	if !certificates.IsCertificateRequestApproved(csr) {
		return nil
	}

	// Fast-path to avoid any additional processing if the CSRs signerName does not match
	if *csr.Spec.SignerName != s.signerName {
		return nil
	}

	x509cr, err := capihelper.ParseCSR(csr.Spec.Request)
	if err != nil {
		return fmt.Errorf("unable to parse csr %q: %v", csr.Name, err)
	}
	if !s.isRequestForSignerFn(x509cr, csr.Spec.Usages, *csr.Spec.SignerName) {
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

// getCSRVerificationFuncForSignerName is a function that provides reliable mapping of signer names to verification so that
// we don't have accidents with wiring at some later date.
func getCSRVerificationFuncForSignerName(signerName string) (isRequestForSignerFunc, error) {
	switch signerName {
	case capi.KubeletServingSignerName:
		return isKubeletServing, nil
	case capi.KubeAPIServerClientKubeletSignerName:
		return isKubeletClient, nil
	case capi.KubeAPIServerClientSignerName:
		return isKubeAPIServerClient, nil
	case capi.LegacyUnknownSignerName:
		return isLegacyUnknown, nil
	default:
		// TODO type this error so that a different reporting loop (one without a signing cert), can mark
		//  CSRs with unknown kube signers as terminal if we wish.  This largely depends on how tightly we want to control
		//  our signerNames.
		return nil, fmt.Errorf("unrecongized signerName: %q", signerName)
	}
}

func isKubeletServing(req *x509.CertificateRequest, usages []capi.KeyUsage, signerName string) bool {
	if signerName != capi.KubeletServingSignerName {
		return false
	}
	return capihelper.IsKubeletServingCSR(req, usages)
}

func isKubeletClient(req *x509.CertificateRequest, usages []capi.KeyUsage, signerName string) bool {
	if signerName != capi.KubeAPIServerClientKubeletSignerName {
		return false
	}
	return capihelper.IsKubeletClientCSR(req, usages)
}

func isKubeAPIServerClient(req *x509.CertificateRequest, usages []capi.KeyUsage, signerName string) bool {
	if signerName != capi.KubeAPIServerClientSignerName {
		return false
	}
	return validAPIServerClientUsages(usages)
}

func isLegacyUnknown(req *x509.CertificateRequest, usages []capi.KeyUsage, signerName string) bool {
	if signerName != capi.LegacyUnknownSignerName {
		return false
	}
	// No restrictions are applied to the legacy-unknown signerName to
	// maintain backward compatibility in v1beta1.
	return true
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
