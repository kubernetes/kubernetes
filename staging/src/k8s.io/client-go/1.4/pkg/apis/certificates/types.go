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
	"k8s.io/client-go/1.4/pkg/api"
	"k8s.io/client-go/1.4/pkg/api/unversioned"
)

// +genclient=true
// +nonNamespaced=true

// Describes a certificate signing request
type CertificateSigningRequest struct {
	unversioned.TypeMeta `json:",inline"`
	api.ObjectMeta       `json:"metadata,omitempty"`

	// The certificate request itself and any additional information.
	Spec CertificateSigningRequestSpec `json:"spec,omitempty"`

	// Derived information about the request.
	Status CertificateSigningRequestStatus `json:"status,omitempty"`
}

// This information is immutable after the request is created. Only the Request
// and ExtraInfo fields can be set on creation, other fields are derived by
// Kubernetes and cannot be modified by users.
type CertificateSigningRequestSpec struct {
	// Base64-encoded PKCS#10 CSR data
	Request []byte `json:"request"`

	// Information about the requesting user (if relevant)
	// See user.Info interface for details
	Username string   `json:"username,omitempty"`
	UID      string   `json:"uid,omitempty"`
	Groups   []string `json:"groups,omitempty"`
}

type CertificateSigningRequestStatus struct {
	// Conditions applied to the request, such as approval or denial.
	Conditions []CertificateSigningRequestCondition `json:"conditions,omitempty"`

	// If request was approved, the controller will place the issued certificate here.
	Certificate []byte `json:"certificate,omitempty"`
}

type RequestConditionType string

// These are the possible conditions for a certificate request.
const (
	CertificateApproved RequestConditionType = "Approved"
	CertificateDenied   RequestConditionType = "Denied"
)

type CertificateSigningRequestCondition struct {
	// request approval state, currently Approved or Denied.
	Type RequestConditionType `json:"type"`
	// brief reason for the request state
	Reason string `json:"reason,omitempty"`
	// human readable message with details about the request state
	Message string `json:"message,omitempty"`
	// timestamp for the last update to this condition
	LastUpdateTime unversioned.Time `json:"lastUpdateTime,omitempty"`
}

type CertificateSigningRequestList struct {
	unversioned.TypeMeta `json:",inline"`
	unversioned.ListMeta `json:"metadata,omitempty"`

	Items []CertificateSigningRequest `json:"items,omitempty"`
}
