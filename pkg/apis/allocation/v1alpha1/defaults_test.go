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

package v1alpha1_test

import (
	"testing"

	"github.com/google/go-cmp/cmp"

	allocation "k8s.io/api/allocation/v1alpha1"

	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"k8s.io/kubernetes/pkg/api/legacyscheme"
	_ "k8s.io/kubernetes/pkg/apis/allocation/install"
)

func TestSetDefaultAddress(t *testing.T) {

	tests := map[string]struct {
		original *allocation.IPAddress
		expected *allocation.IPAddress
	}{
		"should set appropriate defaults IPv4": {
			original: &allocation.IPAddress{
				ObjectMeta: metav1.ObjectMeta{
					Name: "3232235777",
				},
			},
			expected: &allocation.IPAddress{
				ObjectMeta: metav1.ObjectMeta{
					Name: "3232235777",
				},
				Spec: allocation.IPAddressSpec{
					Address: "192.168.1.1",
				},
			},
		},
		"should set appropriate defaults IPv6": {
			original: &allocation.IPAddress{
				ObjectMeta: metav1.ObjectMeta{
					Name: "42540765935913617771317959390390124546",
				},
			},
			expected: &allocation.IPAddress{
				ObjectMeta: metav1.ObjectMeta{
					Name: "42540765935913617771317959390390124546",
				},
				Spec: allocation.IPAddressSpec{
					Address: "2001:db2::2",
				},
			},
		},
	}

	for _, test := range tests {
		actual := test.original
		expected := test.expected
		legacyscheme.Scheme.Default(actual)
		if !apiequality.Semantic.DeepEqual(actual, expected) {
			t.Error(cmp.Diff(expected, actual))
		}
	}
}
