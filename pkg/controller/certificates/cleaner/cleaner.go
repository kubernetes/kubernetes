/*
Copyright 2017 The Kubernetes Authors.

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

// Package cleaner implements an automated cleaner that does garbage collection
// on CSRs that meet specific criteria. With automated CSR requests and
// automated approvals, the volume of CSRs only increases over time, at a rapid
// rate if the certificate duration is short.
package cleaner

import (
	"context"
	"crypto/x509"
	"encoding/pem"
	"fmt"
	"time"

	"k8s.io/klog/v2"

	capi "k8s.io/api/certificates/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	certificatesinformers "k8s.io/client-go/informers/certificates/v1beta1"
	csrclient "k8s.io/client-go/kubernetes/typed/certificates/v1beta1"
	certificateslisters "k8s.io/client-go/listers/certificates/v1beta1"
)

const (
	// The interval to list all CSRs and check each one against the criteria to
	// automatically clean it up.
	pollingInterval = 1 * time.Hour
	// The time periods after which these different CSR statuses should be
	// cleaned up.
	approvedExpiration = 1 * time.Hour
	deniedExpiration   = 1 * time.Hour
	pendingExpiration  = 24 * time.Hour
)

// CSRCleanerController is a controller that garbage collects old certificate
// signing requests (CSRs). Since there are mechanisms that automatically
// create CSRs, and mechanisms that automatically approve CSRs, in order to
// prevent a build up of CSRs over time, it is necessary to GC them. CSRs will
// be removed if they meet one of the following criteria: the CSR is Approved
// with a certificate and is old enough to be past the GC issued deadline, the
// CSR is denied and is old enough to be past the GC denied deadline, the CSR
// is Pending and is old enough to be past the GC pending deadline, the CSR is
// approved with a certificate and the certificate is expired.
type CSRCleanerController struct {
	csrClient csrclient.CertificateSigningRequestInterface
	csrLister certificateslisters.CertificateSigningRequestLister
}

// NewCSRCleanerController creates a new CSRCleanerController.
func NewCSRCleanerController(
	csrClient csrclient.CertificateSigningRequestInterface,
	csrInformer certificatesinformers.CertificateSigningRequestInformer,
) *CSRCleanerController {
	return &CSRCleanerController{
		csrClient: csrClient,
		csrLister: csrInformer.Lister(),
	}
}

// Run the main goroutine responsible for watching and syncing jobs.
func (ccc *CSRCleanerController) Run(workers int, stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()

	klog.Infof("Starting CSR cleaner controller")
	defer klog.Infof("Shutting down CSR cleaner controller")

	for i := 0; i < workers; i++ {
		go wait.Until(ccc.worker, pollingInterval, stopCh)
	}

	<-stopCh
}

// worker runs a thread that dequeues CSRs, handles them, and marks them done.
func (ccc *CSRCleanerController) worker() {
	csrs, err := ccc.csrLister.List(labels.Everything())
	if err != nil {
		klog.Errorf("Unable to list CSRs: %v", err)
		return
	}
	for _, csr := range csrs {
		if err := ccc.handle(csr); err != nil {
			klog.Errorf("Error while attempting to clean CSR %q: %v", csr.Name, err)
		}
	}
}

func (ccc *CSRCleanerController) handle(csr *capi.CertificateSigningRequest) error {
	isIssuedExpired, err := isIssuedExpired(csr)
	if err != nil {
		return err
	}
	if isIssuedPastDeadline(csr) || isDeniedPastDeadline(csr) || isPendingPastDeadline(csr) || isIssuedExpired {
		if err := ccc.csrClient.Delete(context.TODO(), csr.Name, metav1.DeleteOptions{}); err != nil {
			return fmt.Errorf("unable to delete CSR %q: %v", csr.Name, err)
		}
	}
	return nil
}

// isIssuedExpired checks if the CSR has been issued a certificate and if the
// expiration of the certificate (the NotAfter value) has passed.
func isIssuedExpired(csr *capi.CertificateSigningRequest) (bool, error) {
	isExpired, err := isExpired(csr)
	if err != nil {
		return false, err
	}
	for _, c := range csr.Status.Conditions {
		if c.Type == capi.CertificateApproved && isIssued(csr) && isExpired {
			klog.Infof("Cleaning CSR %q as the associated certificate is expired.", csr.Name)
			return true, nil
		}
	}
	return false, nil
}

// isPendingPastDeadline checks if the certificate has a Pending status and the
// creation time of the CSR is passed the deadline that pending requests are
// maintained for.
func isPendingPastDeadline(csr *capi.CertificateSigningRequest) bool {
	// If there are no Conditions on the status, the CSR will appear via
	// `kubectl` as `Pending`.
	if len(csr.Status.Conditions) == 0 && isOlderThan(csr.CreationTimestamp, pendingExpiration) {
		klog.Infof("Cleaning CSR %q as it is more than %v old and unhandled.", csr.Name, pendingExpiration)
		return true
	}
	return false
}

// isDeniedPastDeadline checks if the certificate has a Denied status and the
// creation time of the CSR is passed the deadline that denied requests are
// maintained for.
func isDeniedPastDeadline(csr *capi.CertificateSigningRequest) bool {
	for _, c := range csr.Status.Conditions {
		if c.Type == capi.CertificateDenied && isOlderThan(c.LastUpdateTime, deniedExpiration) {
			klog.Infof("Cleaning CSR %q as it is more than %v old and denied.", csr.Name, deniedExpiration)
			return true
		}
	}
	return false
}

// isIssuedPastDeadline checks if the certificate has an Issued status and the
// creation time of the CSR is passed the deadline that issued requests are
// maintained for.
func isIssuedPastDeadline(csr *capi.CertificateSigningRequest) bool {
	for _, c := range csr.Status.Conditions {
		if c.Type == capi.CertificateApproved && isIssued(csr) && isOlderThan(c.LastUpdateTime, approvedExpiration) {
			klog.Infof("Cleaning CSR %q as it is more than %v old and approved.", csr.Name, approvedExpiration)
			return true
		}
	}
	return false
}

// isOlderThan checks that t is a non-zero time after time.Now() + d.
func isOlderThan(t metav1.Time, d time.Duration) bool {
	return !t.IsZero() && t.Sub(time.Now()) < -1*d
}

// isIssued checks if the CSR has `Issued` status. There is no explicit
// 'Issued' status. Implicitly, if there is a certificate associated with the
// CSR, the CSR statuses that are visible via `kubectl` will include 'Issued'.
func isIssued(csr *capi.CertificateSigningRequest) bool {
	return csr.Status.Certificate != nil
}

// isExpired checks if the CSR has a certificate and the date in the `NotAfter`
// field has gone by.
func isExpired(csr *capi.CertificateSigningRequest) (bool, error) {
	if csr.Status.Certificate == nil {
		return false, nil
	}
	block, _ := pem.Decode(csr.Status.Certificate)
	if block == nil {
		return false, fmt.Errorf("expected the certificate associated with the CSR to be PEM encoded")
	}
	certs, err := x509.ParseCertificates(block.Bytes)
	if err != nil {
		return false, fmt.Errorf("unable to parse certificate data: %v", err)
	}
	return time.Now().After(certs[0].NotAfter), nil
}
