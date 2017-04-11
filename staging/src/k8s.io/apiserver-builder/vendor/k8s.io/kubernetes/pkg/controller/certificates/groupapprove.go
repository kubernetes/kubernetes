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

package certificates

import (
	"fmt"
	"reflect"
	"strings"

	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	certificates "k8s.io/kubernetes/pkg/apis/certificates/v1beta1"
)

// groupApprover implements AutoApprover for signing Kubelet certificates.
type groupApprover struct {
	approveAllKubeletCSRsForGroup string
}

// NewGroupApprover creates an approver that accepts any CSR requests where the subject group contains approveAllKubeletCSRsForGroup.
func NewGroupApprover(approveAllKubeletCSRsForGroup string) AutoApprover {
	return &groupApprover{
		approveAllKubeletCSRsForGroup: approveAllKubeletCSRsForGroup,
	}
}

func (cc *groupApprover) AutoApprove(csr *certificates.CertificateSigningRequest) (*certificates.CertificateSigningRequest, error) {
	// short-circuit if we're not auto-approving
	if cc.approveAllKubeletCSRsForGroup == "" {
		return csr, nil
	}
	// short-circuit if we're already approved or denied
	if approved, denied := getCertApprovalCondition(&csr.Status); approved || denied {
		return csr, nil
	}

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

	x509cr, err := certificates.ParseCSR(csr)
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

	csr.Status.Conditions = append(csr.Status.Conditions, certificates.CertificateSigningRequestCondition{
		Type:    certificates.CertificateApproved,
		Reason:  "AutoApproved",
		Message: "Auto approving of all kubelet CSRs is enabled on the controller manager",
	})
	return csr, nil
}

var kubeletClientUsages = []certificates.KeyUsage{
	certificates.UsageKeyEncipherment,
	certificates.UsageDigitalSignature,
	certificates.UsageClientAuth,
}

func hasExactUsages(csr *certificates.CertificateSigningRequest, usages []certificates.KeyUsage) bool {
	if len(usages) != len(csr.Spec.Usages) {
		return false
	}

	usageMap := map[certificates.KeyUsage]struct{}{}
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
