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

import (
	"testing"
)

func TestCertKeyContentEquals(t *testing.T) {
	tests := []struct {
		name     string
		lhs      *certKeyContent
		rhs      *certKeyContent
		expected bool
	}{
		{
			name:     "both nil",
			expected: true,
		},
		{
			name:     "lhs nil",
			rhs:      &certKeyContent{},
			expected: false,
		},
		{
			name:     "rhs nil",
			lhs:      &certKeyContent{},
			expected: false,
		},
		{
			name:     "same",
			lhs:      &certKeyContent{cert: []byte("foo"), key: []byte("baz")},
			rhs:      &certKeyContent{cert: []byte("foo"), key: []byte("baz")},
			expected: true,
		},
		{
			name:     "different cert",
			lhs:      &certKeyContent{cert: []byte("foo"), key: []byte("baz")},
			rhs:      &certKeyContent{cert: []byte("bar"), key: []byte("baz")},
			expected: false,
		},
		{
			name:     "different key",
			lhs:      &certKeyContent{cert: []byte("foo"), key: []byte("baz")},
			rhs:      &certKeyContent{cert: []byte("foo"), key: []byte("qux")},
			expected: false,
		},
		{
			name:     "different cert and key",
			lhs:      &certKeyContent{cert: []byte("foo"), key: []byte("baz")},
			rhs:      &certKeyContent{cert: []byte("bar"), key: []byte("qux")},
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

func TestSNICertKeyContentEquals(t *testing.T) {
	tests := []struct {
		name     string
		lhs      *sniCertKeyContent
		rhs      *sniCertKeyContent
		expected bool
	}{
		{
			name:     "both nil",
			expected: true,
		},
		{
			name:     "lhs nil",
			rhs:      &sniCertKeyContent{},
			expected: false,
		},
		{
			name:     "rhs nil",
			lhs:      &sniCertKeyContent{},
			expected: false,
		},
		{
			name:     "same",
			lhs:      &sniCertKeyContent{certKeyContent: certKeyContent{cert: []byte("foo"), key: []byte("baz")}, sniNames: []string{"a"}},
			rhs:      &sniCertKeyContent{certKeyContent: certKeyContent{cert: []byte("foo"), key: []byte("baz")}, sniNames: []string{"a"}},
			expected: true,
		},
		{
			name:     "different cert",
			lhs:      &sniCertKeyContent{certKeyContent: certKeyContent{cert: []byte("foo"), key: []byte("baz")}, sniNames: []string{"a"}},
			rhs:      &sniCertKeyContent{certKeyContent: certKeyContent{cert: []byte("bar"), key: []byte("baz")}, sniNames: []string{"a"}},
			expected: false,
		},
		{
			name:     "different key",
			lhs:      &sniCertKeyContent{certKeyContent: certKeyContent{cert: []byte("foo"), key: []byte("baz")}, sniNames: []string{"a"}},
			rhs:      &sniCertKeyContent{certKeyContent: certKeyContent{cert: []byte("foo"), key: []byte("qux")}, sniNames: []string{"a"}},
			expected: false,
		},
		{
			name:     "different cert and key",
			lhs:      &sniCertKeyContent{certKeyContent: certKeyContent{cert: []byte("foo"), key: []byte("baz")}, sniNames: []string{"a"}},
			rhs:      &sniCertKeyContent{certKeyContent: certKeyContent{cert: []byte("bar"), key: []byte("qux")}, sniNames: []string{"a"}},
			expected: false,
		},
		{
			name:     "different names",
			lhs:      &sniCertKeyContent{certKeyContent: certKeyContent{cert: []byte("foo"), key: []byte("baz")}, sniNames: []string{"a"}},
			rhs:      &sniCertKeyContent{certKeyContent: certKeyContent{cert: []byte("foo"), key: []byte("baz")}, sniNames: []string{"b"}},
			expected: false,
		},
		{
			name:     "extra names",
			lhs:      &sniCertKeyContent{certKeyContent: certKeyContent{cert: []byte("foo"), key: []byte("baz")}, sniNames: []string{"a"}},
			rhs:      &sniCertKeyContent{certKeyContent: certKeyContent{cert: []byte("foo"), key: []byte("baz")}, sniNames: []string{"a", "b"}},
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
