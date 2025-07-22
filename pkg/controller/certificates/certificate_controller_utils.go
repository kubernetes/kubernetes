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
	certificates "k8s.io/api/certificates/v1"
	v1 "k8s.io/api/core/v1"
)

// IsCertificateRequestApproved returns true if a certificate request has the
// "Approved" condition and no "Denied" conditions; false otherwise.
func IsCertificateRequestApproved(csr *certificates.CertificateSigningRequest) bool {
	approved, denied := GetCertApprovalCondition(&csr.Status)
	return approved && !denied
}

// HasTrueCondition returns true if the csr contains a condition of the specified type with a status that is set to True or is empty
func HasTrueCondition(csr *certificates.CertificateSigningRequest, conditionType certificates.RequestConditionType) bool {
	for _, c := range csr.Status.Conditions {
		if c.Type == conditionType && (len(c.Status) == 0 || c.Status == v1.ConditionTrue) {
			return true
		}
	}
	return false
}

func GetCertApprovalCondition(status *certificates.CertificateSigningRequestStatus) (approved bool, denied bool) {
	for _, c := range status.Conditions {
		if c.Type == certificates.CertificateApproved {
			approved = true
		}
		if c.Type == certificates.CertificateDenied {
			denied = true
		}
	}
	return
}
