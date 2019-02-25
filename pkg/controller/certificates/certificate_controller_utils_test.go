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
	"github.com/stretchr/testify/assert"
	"k8s.io/api/certificates/v1beta1"
	"k8s.io/apimachinery/pkg/apis/meta/v1"
	"testing"
)

func TestIsCertificateRequestApproved(t *testing.T) {
	testCases := []struct {
		name               string
		conditions         []v1beta1.CertificateSigningRequestCondition
		expectedIsApproved bool
	}{
		{
			"Not any conditions exist",
			nil,
			false,
		}, {
			"Approved not exist and Denied exist",
			[]v1beta1.CertificateSigningRequestCondition{
				{
					Type: v1beta1.CertificateDenied,
				},
			},
			false,
		}, {
			"Approved exist and Denied not exist",
			[]v1beta1.CertificateSigningRequestCondition{
				{
					Type: v1beta1.CertificateApproved,
				},
			},
			true,
		}, {
			"Both of Approved and Denied exist",
			[]v1beta1.CertificateSigningRequestCondition{
				{
					Type: v1beta1.CertificateApproved,
				},
				{
					Type: v1beta1.CertificateDenied,
				},
			},
			false,
		},
	}

	for _, tc := range testCases {
		csr := &v1beta1.CertificateSigningRequest{
			ObjectMeta: v1.ObjectMeta{
				Name: "fake-csr",
			},
			Status: v1beta1.CertificateSigningRequestStatus{
				Conditions: tc.conditions,
			},
		}

		assert.Equalf(t, tc.expectedIsApproved, IsCertificateRequestApproved(csr), "Failed to test: %s", tc.name)
	}
}
