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

// Package approver implements an automated approver for kubelet certificates.
package approver

import (
	"context"
	"crypto/x509"
	"fmt"

	authorization "k8s.io/api/authorization/v1"
	capi "k8s.io/api/certificates/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	certificatesinformers "k8s.io/client-go/informers/certificates/v1beta1"
	clientset "k8s.io/client-go/kubernetes"

	capihelper "k8s.io/kubernetes/pkg/apis/certificates/v1beta1"
	"k8s.io/kubernetes/pkg/controller/certificates"
)

type csrRecognizer struct {
	recognize      func(csr *capi.CertificateSigningRequest, x509cr *x509.CertificateRequest) bool
	permission     authorization.ResourceAttributes
	successMessage string
}

type sarApprover struct {
	client      clientset.Interface
	recognizers []csrRecognizer
}

// NewCSRApprovingController creates a new CSRApprovingController.
func NewCSRApprovingController(client clientset.Interface, csrInformer certificatesinformers.CertificateSigningRequestInformer) *certificates.CertificateController {
	approver := &sarApprover{
		client:      client,
		recognizers: recognizers(),
	}
	return certificates.NewCertificateController(
		"csrapproving",
		client,
		csrInformer,
		approver.handle,
	)
}

func recognizers() []csrRecognizer {
	recognizers := []csrRecognizer{
		{
			recognize:      isSelfNodeClientCert,
			permission:     authorization.ResourceAttributes{Group: "certificates.k8s.io", Resource: "certificatesigningrequests", Verb: "create", Subresource: "selfnodeclient"},
			successMessage: "Auto approving self kubelet client certificate after SubjectAccessReview.",
		},
		{
			recognize:      isNodeClientCert,
			permission:     authorization.ResourceAttributes{Group: "certificates.k8s.io", Resource: "certificatesigningrequests", Verb: "create", Subresource: "nodeclient"},
			successMessage: "Auto approving kubelet client certificate after SubjectAccessReview.",
		},
	}
	return recognizers
}

func (a *sarApprover) handle(csr *capi.CertificateSigningRequest) error {
	if len(csr.Status.Certificate) != 0 {
		return nil
	}
	if approved, denied := certificates.GetCertApprovalCondition(&csr.Status); approved || denied {
		return nil
	}
	x509cr, err := capihelper.ParseCSR(csr.Spec.Request)
	if err != nil {
		return fmt.Errorf("unable to parse csr %q: %v", csr.Name, err)
	}

	tried := []string{}

	for _, r := range a.recognizers {
		if !r.recognize(csr, x509cr) {
			continue
		}

		tried = append(tried, r.permission.Subresource)

		approved, err := a.authorize(csr, r.permission)
		if err != nil {
			return err
		}
		if approved {
			appendApprovalCondition(csr, r.successMessage)
			_, err = a.client.CertificatesV1beta1().CertificateSigningRequests().UpdateApproval(context.Background(), csr, metav1.UpdateOptions{})
			if err != nil {
				return fmt.Errorf("error updating approval for csr: %v", err)
			}
			return nil
		}
	}

	if len(tried) != 0 {
		return certificates.IgnorableError("recognized csr %q as %v but subject access review was not approved", csr.Name, tried)
	}

	return nil
}

func (a *sarApprover) authorize(csr *capi.CertificateSigningRequest, rattrs authorization.ResourceAttributes) (bool, error) {
	extra := make(map[string]authorization.ExtraValue)
	for k, v := range csr.Spec.Extra {
		extra[k] = authorization.ExtraValue(v)
	}

	sar := &authorization.SubjectAccessReview{
		Spec: authorization.SubjectAccessReviewSpec{
			User:               csr.Spec.Username,
			UID:                csr.Spec.UID,
			Groups:             csr.Spec.Groups,
			Extra:              extra,
			ResourceAttributes: &rattrs,
		},
	}
	sar, err := a.client.AuthorizationV1().SubjectAccessReviews().Create(context.TODO(), sar, metav1.CreateOptions{})
	if err != nil {
		return false, err
	}
	return sar.Status.Allowed, nil
}

func appendApprovalCondition(csr *capi.CertificateSigningRequest, message string) {
	csr.Status.Conditions = append(csr.Status.Conditions, capi.CertificateSigningRequestCondition{
		Type:    capi.CertificateApproved,
		Reason:  "AutoApproved",
		Message: message,
	})
}

func isNodeClientCert(csr *capi.CertificateSigningRequest, x509cr *x509.CertificateRequest) bool {
	isClientCSR := capihelper.IsKubeletClientCSR(x509cr, csr.Spec.Usages)
	if !isClientCSR {
		return false
	}
	return *csr.Spec.SignerName == capi.KubeAPIServerClientKubeletSignerName
}

func isSelfNodeClientCert(csr *capi.CertificateSigningRequest, x509cr *x509.CertificateRequest) bool {
	if !isNodeClientCert(csr, x509cr) {
		return false
	}
	if csr.Spec.Username != x509cr.Subject.CommonName {
		return false
	}
	return true
}
