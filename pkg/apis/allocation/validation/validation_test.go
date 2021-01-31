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

package validation

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/apis/allocation"
)

func TestValidateIPRange(t *testing.T) {

	testCases := map[string]struct {
		expectedErrors int
		ipRange        *allocation.IPRange
	}{
		"empty-iprange": {
			expectedErrors: 1,
			ipRange: &allocation.IPRange{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "hello",
					Namespace: "world",
				},
			},
		},
		"good-iprange-ipv4": {
			expectedErrors: 0,
			ipRange: &allocation.IPRange{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "hello",
					Namespace: "world",
				},
				Spec: allocation.IPRangeSpec{
					Range: "192.168.0.0/24",
				},
			},
		},
		"good-iprange-ipv6": {
			expectedErrors: 0,
			ipRange: &allocation.IPRange{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "hello",
					Namespace: "world",
				},
				Spec: allocation.IPRangeSpec{
					Range: "fd00:1234::/64",
				},
			},
		},
		"not-iprange-ipv4": {
			expectedErrors: 1,
			ipRange: &allocation.IPRange{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "hello",
					Namespace: "world",
				},
				Spec: allocation.IPRangeSpec{
					Range: "sadasdsad",
				},
			},
		},
		"iponly-iprange-ipv4": {
			expectedErrors: 1,
			ipRange: &allocation.IPRange{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "hello",
					Namespace: "world",
				},
				Spec: allocation.IPRangeSpec{
					Range: "192.168.0.1",
				},
			},
		},
		"badip-iprange-ipv4": {
			expectedErrors: 1,
			ipRange: &allocation.IPRange{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "hello",
					Namespace: "world",
				},
				Spec: allocation.IPRangeSpec{
					Range: "192.168.0.1/24",
				},
			},
		},
		"badip-iprange-ipv6": {
			expectedErrors: 1,
			ipRange: &allocation.IPRange{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "hello",
					Namespace: "world",
				},
				Spec: allocation.IPRangeSpec{
					Range: "fd00:1234::2/64",
				},
			},
		},
	}

	for name, testCase := range testCases {
		t.Run(name, func(t *testing.T) {
			errs := ValidateIPRange(testCase.ipRange)
			if len(errs) != testCase.expectedErrors {
				t.Errorf("Expected %d errors, got %d errors: %v", testCase.expectedErrors, len(errs), errs)
			}
		})
	}
}

func TestValidateIPAddress(t *testing.T) {
	testCases := map[string]struct {
		expectedErrors int
		ipAddress      *allocation.IPAddress
	}{
		"good ipv4": {
			expectedErrors: 0,
			ipAddress: &allocation.IPAddress{
				ObjectMeta: metav1.ObjectMeta{
					Name: "1.1.1.1",
				},
			},
		},
		"good ipv6 short format": {
			expectedErrors: 0,
			ipAddress: &allocation.IPAddress{
				ObjectMeta: metav1.ObjectMeta{
					Name: "2001::1",
				},
			},
		},
		"good ipv6 long format": {
			expectedErrors: 0,
			ipAddress: &allocation.IPAddress{
				ObjectMeta: metav1.ObjectMeta{
					Name: "2001:0db8:0000:0000:0000:0000:0000:0001",
				},
			},
		},
		"wrong-ipaddress-nonamespce": {
			expectedErrors: 1,
			ipAddress: &allocation.IPAddress{
				ObjectMeta: metav1.ObjectMeta{
					Name: "asdasdsdads",
				},
			},
		},
		"wrong-ipaddress-namespace": {
			expectedErrors: 2,
			ipAddress: &allocation.IPAddress{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "3232235777",
					Namespace: "world",
				},
			},
		},
	}

	for name, testCase := range testCases {
		t.Run(name, func(t *testing.T) {
			errs := ValidateIPAddress(testCase.ipAddress)
			if len(errs) != testCase.expectedErrors {
				t.Errorf("Expected %d errors, got %d errors: %v", testCase.expectedErrors, len(errs), errs)
			}
		})
	}
}
