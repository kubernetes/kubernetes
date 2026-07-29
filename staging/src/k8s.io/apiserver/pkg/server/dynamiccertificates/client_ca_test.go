/*
Copyright 2019 The Kubernetes Authors.

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

package dynamiccertificates

import "testing"

func TestDynamicCertificateContentEquals(t *testing.T) {
	tests := []struct {
		name     string
		lhs      *dynamicCertificateContent
		rhs      *dynamicCertificateContent
		expected bool
	}{
		{
			name:     "both nil",
			expected: true,
		},
		{
			name:     "lhs nil",
			rhs:      &dynamicCertificateContent{},
			expected: false,
		},
		{
			name:     "rhs nil",
			lhs:      &dynamicCertificateContent{},
			expected: false,
		},
		{
			name: "same",
			lhs: &dynamicCertificateContent{
				clientCA: caBundleContent{caBundle: []byte("foo")},
			},
			rhs: &dynamicCertificateContent{
				clientCA: caBundleContent{caBundle: []byte("foo")},
			},
			expected: true,
		},
		{
			name: "different",
			lhs: &dynamicCertificateContent{
				clientCA: caBundleContent{caBundle: []byte("foo")},
			},
			rhs: &dynamicCertificateContent{
				clientCA: caBundleContent{caBundle: []byte("bar")},
			},
			expected: false,
		},
		{
			name: "same with serving",
			lhs: &dynamicCertificateContent{
				clientCA:    caBundleContent{caBundle: []byte("foo")},
				servingCert: certKeyContent{cert: []byte("foo"), key: []byte("foo")},
			},
			rhs: &dynamicCertificateContent{
				clientCA:    caBundleContent{caBundle: []byte("foo")},
				servingCert: certKeyContent{cert: []byte("foo"), key: []byte("foo")},
			},
			expected: true,
		},
		{
			name: "different serving cert",
			lhs: &dynamicCertificateContent{
				clientCA:    caBundleContent{caBundle: []byte("foo")},
				servingCert: certKeyContent{cert: []byte("foo"), key: []byte("foo")},
			},
			rhs: &dynamicCertificateContent{
				clientCA:    caBundleContent{caBundle: []byte("foo")},
				servingCert: certKeyContent{cert: []byte("bar"), key: []byte("foo")},
			},
			expected: false,
		},
		{
			name: "different serving key",
			lhs: &dynamicCertificateContent{
				clientCA:    caBundleContent{caBundle: []byte("foo")},
				servingCert: certKeyContent{cert: []byte("foo"), key: []byte("foo")},
			},
			rhs: &dynamicCertificateContent{
				clientCA:    caBundleContent{caBundle: []byte("foo")},
				servingCert: certKeyContent{cert: []byte("foo"), key: []byte("bar")},
			},
			expected: false,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			actual := test.lhs.Equal(test.rhs)
			if actual != test.expected {
				t.Error(actual)
			}
		})
	}
}

func TestCABundleContentEquals(t *testing.T) {
	tests := []struct {
		name     string
		lhs      *caBundleContent
		rhs      *caBundleContent
		expected bool
	}{
		{
			name:     "both nil",
			expected: true,
		},
		{
			name:     "lhs nil",
			rhs:      &caBundleContent{},
			expected: false,
		},
		{
			name:     "rhs nil",
			lhs:      &caBundleContent{},
			expected: false,
		},
		{
			name:     "same",
			lhs:      &caBundleContent{caBundle: []byte("foo")},
			rhs:      &caBundleContent{caBundle: []byte("foo")},
			expected: true,
		},
		{
			name:     "different",
			lhs:      &caBundleContent{caBundle: []byte("foo")},
			rhs:      &caBundleContent{caBundle: []byte("bar")},
			expected: false,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			actual := test.lhs.Equal(test.rhs)
			if actual != test.expected {
				t.Error(actual)
			}
		})
	}
}
