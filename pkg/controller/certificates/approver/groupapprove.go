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
	"fmt"
	"reflect"
	"strings"

	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	capi "k8s.io/kubernetes/pkg/apis/certificates/v1beta1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	certificatesinformers "k8s.io/kubernetes/pkg/client/informers/informers_generated/externalversions/certificates/v1beta1"
	"k8s.io/kubernetes/pkg/controller/certificates"
)

func NewCSRApprovingController(
	client clientset.Interface,
	csrInformer certificatesinformers.CertificateSigningRequestInformer,
	approveAllKubeletCSRsForGroup string,
) (*certificates.CertificateController, error) {
	approver := &groupApprover{
		approveAllKubeletCSRsForGroup: approveAllKubeletCSRsForGroup,
		client: client,
	}
	return certificates.NewCertificateController(
		client,
		csrInformer,
		approver.handle,
	)
}

// groupApprover implements AutoApprover for signing Kubelet certificates.
type groupApprover struct {
	approveAllKubeletCSRsForGroup string
	client                        clientset.Interface
}

func (ga *groupApprover) handle(csr *capi.CertificateSigningRequest) error {
	// short-circuit if we're already approved or denied
	if approved, denied := certificates.GetCertApprovalCondition(&csr.Status); approved || denied {
		return nil
	}
	csr, err := ga.autoApprove(csr)
	if err != nil {
		return fmt.Errorf("error auto approving csr: %v", err)
	}
	_, err = ga.client.Certificates().CertificateSigningRequests().UpdateApproval(csr)
	if err != nil {
		return fmt.Errorf("error updating approval for csr: %v", err)
	}
	return nil
}

func (cc *groupApprover) autoApprove(csr *capi.CertificateSigningRequest) (*capi.CertificateSigningRequest, error) {
	isKubeletBootstrapGroup := false
	for _, g := range csr.Spec.Groups {
		if g == cc.approveAllKubeletCSRsForGroup {
			isKubeletBootstrapGroup = true
			break
		}
	}
	if !isKubeletBootstrapGroup {
		return csr, nil
	}

	x509cr, err := capi.ParseCSR(csr)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("unable to parse csr %q: %v", csr.Name, err))
		return csr, nil
	}
	if !reflect.DeepEqual([]string{"system:nodes"}, x509cr.Subject.Organization) {
		return csr, nil
	}
	if !strings.HasPrefix(x509cr.Subject.CommonName, "system:node:") {
		return csr, nil
	}
	if len(x509cr.DNSNames)+len(x509cr.EmailAddresses)+len(x509cr.IPAddresses) != 0 {
		return csr, nil
	}
	if !hasExactUsages(csr, kubeletClientUsages) {
		return csr, nil
	}

	csr.Status.Conditions = append(csr.Status.Conditions, capi.CertificateSigningRequestCondition{
		Type:    capi.CertificateApproved,
		Reason:  "AutoApproved",
		Message: "Auto approving of all kubelet CSRs is enabled on the controller manager",
	})
	return csr, nil
}

var kubeletClientUsages = []capi.KeyUsage{
	capi.UsageKeyEncipherment,
	capi.UsageDigitalSignature,
	capi.UsageClientAuth,
}

func hasExactUsages(csr *capi.CertificateSigningRequest, usages []capi.KeyUsage) bool {
	if len(usages) != len(csr.Spec.Usages) {
		return false
	}

	usageMap := map[capi.KeyUsage]struct{}{}
	for _, u := range usages {
		usageMap[u] = struct{}{}
	}

	for _, u := range csr.Spec.Usages {
		if _, ok := usageMap[u]; !ok {
			return false
		}
	}

	return true
}
