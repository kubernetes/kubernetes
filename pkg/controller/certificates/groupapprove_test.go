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

package certificates

import (
	"testing"

	certificates "k8s.io/kubernetes/pkg/apis/certificates/v1beta1"
)

func TestHasKubeletUsages(t *testing.T) {
	cases := []struct {
		usages   []certificates.KeyUsage
		expected bool
	}{
		{
			usages:   nil,
			expected: false,
		},
		{
			usages:   []certificates.KeyUsage{},
			expected: false,
		},
		{
			usages: []certificates.KeyUsage{
				certificates.UsageKeyEncipherment,
				certificates.UsageDigitalSignature,
			},
			expected: false,
		},
		{
			usages: []certificates.KeyUsage{
				certificates.UsageKeyEncipherment,
				certificates.UsageDigitalSignature,
				certificates.UsageServerAuth,
			},
			expected: false,
		},
		{
			usages: []certificates.KeyUsage{
				certificates.UsageKeyEncipherment,
				certificates.UsageDigitalSignature,
				certificates.UsageClientAuth,
			},
			expected: true,
		},
	}
	for _, c := range cases {
		if hasExactUsages(&certificates.CertificateSigningRequest{
			Spec: certificates.CertificateSigningRequestSpec{
				Usages: c.usages,
			},
		}, kubeletClientUsages) != c.expected {
			t.Errorf("unexpected result of hasKubeletUsages(%v), expecting: %v", c.usages, c.expected)
		}
	}
}
