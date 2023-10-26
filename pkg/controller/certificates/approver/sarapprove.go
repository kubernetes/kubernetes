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
	capi "k8s.io/api/certificates/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	certificatesinformers "k8s.io/client-go/informers/certificates/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/component-helpers/kubernetesx509"
	capihelper "k8s.io/kubernetes/pkg/apis/certificates"
	"k8s.io/kubernetes/pkg/controller/certificates"
)

type sarApprover struct {
	client clientset.Interface
}

// NewCSRApprovingController creates a new CSRApprovingController.
func NewCSRApprovingController(ctx context.Context, client clientset.Interface, csrInformer certificatesinformers.CertificateSigningRequestInformer) *certificates.CertificateController {
	approver := &sarApprover{
		client: client,
	}
	return certificates.NewCertificateController(
		ctx,
		"csrapproving",
		client,
		csrInformer,
		approver.handle,
	)
}

func (a *sarApprover) handle(ctx context.Context, csr *capi.CertificateSigningRequest) error {
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

	if isSelfNodeClientCert(csr, x509cr) {
		permission := authorization.ResourceAttributes{
			Group:       "certificates.k8s.io",
			Resource:    "certificatesigningrequests",
			Verb:        "create",
			Subresource: "selfnodeclient",
			Version:     "*",
		}
		approved, err := a.authorize(ctx, csr, permission)
		if err != nil {
			return fmt.Errorf("while authorizing csr %q: %w", csr.ObjectMeta.Name, err)
		}
		if !approved {
			return certificates.IgnorableError("recognized csr %q as selfNodeClientCert but subject access review was not approved", csr.Name)
		}

		appendApprovalCondition(csr, "Auto approving self kubelet client certificate after SubjectAccessReview.")
		_, err = a.client.CertificatesV1().CertificateSigningRequests().UpdateApproval(ctx, csr.Name, csr, metav1.UpdateOptions{})
		if err != nil {
			return fmt.Errorf("error updating approval for csr: %v", err)
		}
		return nil
	}

	if isNodeClientCert(csr, x509cr) {
		permission := authorization.ResourceAttributes{
			Group:       "certificates.k8s.io",
			Resource:    "certificatesigningrequests",
			Verb:        "create",
			Subresource: "nodeclient",
			Version:     "*",
		}
		approved, err := a.authorize(ctx, csr, permission)
		if err != nil {
			return fmt.Errorf("while authorizing csr %q: %w", csr.ObjectMeta.Name, err)
		}
		if !approved {
			return certificates.IgnorableError("recognized csr %q as nodeClientCert but subject access review was not approved", csr.Name)
		}

		appendApprovalCondition(csr, "Auto approving kubelet client certificate after SubjectAccessReview.")
		_, err = a.client.CertificatesV1().CertificateSigningRequests().UpdateApproval(ctx, csr.Name, csr, metav1.UpdateOptions{})
		if err != nil {
			return fmt.Errorf("error updating approval for csr: %v", err)
		}
		return nil

	}

	if csr.Spec.SignerName == capi.KubeAPIServerClientPodSignerName {
		pi, err := kubernetesx509.PodIdentityFromCertificateRequest(x509cr)
		if err != nil {
			return fmt.Errorf("while checking for Kubernetes X.509 extensions: %w", err)
		}
		if pi == nil {
			return fmt.Errorf("CSR addressed to signer %s did not contain a Kubernetes PodIdentity X509 extension", capi.KubeAPIServerClientKubeletSignerName)
		}

		// TODO(KEP-PodCertificates): Adjust the comment below once it's clear
		// exactly which admission controller enforces this.

		// The noderestriction admission controller will have already ensured
		// that the CSR requester is a node/kubelet, and it's actually running the pod
		// referenced in the pod identity extension.  We can just approve the
		// CSR.

		appendApprovalCondition(csr, "Auto approving pod client certificate.")
		_, err = a.client.CertificatesV1().CertificateSigningRequests().UpdateApproval(ctx, csr.Name, csr, metav1.UpdateOptions{})
		if err != nil {
			return fmt.Errorf("while updating approval for csr: %v", err)
		}

		return nil
	}

	// We didn't recognize the CSR as one of the types we auto-approve.  Take no
	// action.  Maybe someone will manually approve it.
	return nil
}

func (a *sarApprover) authorize(ctx context.Context, csr *capi.CertificateSigningRequest, rattrs authorization.ResourceAttributes) (bool, error) {
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
	sar, err := a.client.AuthorizationV1().SubjectAccessReviews().Create(ctx, sar, metav1.CreateOptions{})
	if err != nil {
		return false, err
	}
	return sar.Status.Allowed, nil
}

func appendApprovalCondition(csr *capi.CertificateSigningRequest, message string) {
	csr.Status.Conditions = append(csr.Status.Conditions, capi.CertificateSigningRequestCondition{
		Type:    capi.CertificateApproved,
		Status:  corev1.ConditionTrue,
		Reason:  "AutoApproved",
		Message: message,
	})
}

func isNodeClientCert(csr *capi.CertificateSigningRequest, x509cr *x509.CertificateRequest) bool {
	if csr.Spec.SignerName != capi.KubeAPIServerClientKubeletSignerName {
		return false
	}
	return capihelper.IsKubeletClientCSR(x509cr, usagesToSet(csr.Spec.Usages))
}

func isSelfNodeClientCert(csr *capi.CertificateSigningRequest, x509cr *x509.CertificateRequest) bool {
	if csr.Spec.Username != x509cr.Subject.CommonName {
		return false
	}
	return isNodeClientCert(csr, x509cr)
}

func usagesToSet(usages []capi.KeyUsage) sets.String {
	result := sets.NewString()
	for _, usage := range usages {
		result.Insert(string(usage))
	}
	return result
}
