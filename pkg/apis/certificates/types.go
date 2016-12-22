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
	"k8s.io/kubernetes/pkg/api"
	metav1 "k8s.io/kubernetes/pkg/apis/meta/v1"
)

// +genclient=true
// +nonNamespaced=true

// Describes a certificate signing request
type CertificateSigningRequest struct {
	metav1.TypeMeta
	// +optional
	api.ObjectMeta

	// The certificate request itself and any additional information.
	// +optional
	Spec CertificateSigningRequestSpec

	// Derived information about the request.
	// +optional
	Status CertificateSigningRequestStatus
}

// This information is immutable after the request is created. Only the Request
// and ExtraInfo fields can be set on creation, other fields are derived by
// Kubernetes and cannot be modified by users.
type CertificateSigningRequestSpec struct {
	// Base64-encoded PKCS#10 CSR data
	Request []byte

	// Information about the requesting user (if relevant)
	// See user.Info interface for details
	// +optional
	Username string
	// +optional
	UID string
	// +optional
	Groups []string
}

type CertificateSigningRequestStatus struct {
	// Conditions applied to the request, such as approval or denial.
	// +optional
	Conditions []CertificateSigningRequestCondition

	// If request was approved, the controller will place the issued certificate here.
	// +optional
	Certificate []byte
}

type RequestConditionType string

// These are the possible conditions for a certificate request.
const (
	CertificateApproved RequestConditionType = "Approved"
	CertificateDenied   RequestConditionType = "Denied"
)

type CertificateSigningRequestCondition struct {
	// request approval state, currently Approved or Denied.
	Type RequestConditionType
	// brief reason for the request state
	// +optional
	Reason string
	// human readable message with details about the request state
	// +optional
	Message string
	// timestamp for the last update to this condition
	// +optional
	LastUpdateTime metav1.Time
}

type CertificateSigningRequestList struct {
	metav1.TypeMeta
	// +optional
	metav1.ListMeta

	// +optional
	Items []CertificateSigningRequest
}
