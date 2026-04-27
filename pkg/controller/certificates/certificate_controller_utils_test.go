/*
Copyright 2018 The Kubernetes Authors.

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
	"testing"

	"github.com/stretchr/testify/assert"

	certificatesapi "k8s.io/api/certificates/v1"
	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestIsCertificateRequestApproved(t *testing.T) {
	testCases := []struct {
		name               string
		conditions         []certificatesapi.CertificateSigningRequestCondition
		expectedIsApproved bool
	}{
		{
			"Not any conditions exist",
			nil,
			false,
		}, {
			"Approved not exist and Denied exist",
			[]certificatesapi.CertificateSigningRequestCondition{
				{
					Type: certificatesapi.CertificateDenied,
				},
			},
			false,
		}, {
			"Approved exist and Denied not exist",
			[]certificatesapi.CertificateSigningRequestCondition{
				{
					Type: certificatesapi.CertificateApproved,
				},
			},
			true,
		}, {
			"Both of Approved and Denied exist",
			[]certificatesapi.CertificateSigningRequestCondition{
				{
					Type: certificatesapi.CertificateApproved,
				},
				{
					Type: certificatesapi.CertificateDenied,
				},
			},
			false,
		},
	}

	for _, tc := range testCases {
		csr := &certificatesapi.CertificateSigningRequest{
			ObjectMeta: v1.ObjectMeta{
				Name: "fake-csr",
			},
			Status: certificatesapi.CertificateSigningRequestStatus{
				Conditions: tc.conditions,
			},
		}

		assert.Equalf(t, tc.expectedIsApproved, IsCertificateRequestApproved(csr), "Failed to test: %s", tc.name)
	}
}
