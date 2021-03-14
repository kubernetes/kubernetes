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

func TestValidateIPRequest(t *testing.T) {

	testCases := map[string]struct {
		expectedErrors int
		ipRequest      *allocation.IPRequest
	}{
		"empty-iprequest": {
			expectedErrors: 1,
			ipRequest: &allocation.IPRequest{
				ObjectMeta: metav1.ObjectMeta{
					Name: "",
				},
			},
		},
		"good-iprequest-ipv4": {
			expectedErrors: 0,
			ipRequest: &allocation.IPRequest{
				ObjectMeta: metav1.ObjectMeta{
					Name: "192.168.2.2",
				},
			},
		},
		"good-iprequest-ipv6": {
			expectedErrors: 0,
			ipRequest: &allocation.IPRequest{
				ObjectMeta: metav1.ObjectMeta{
					Name: "2001:db2::2",
				},
			},
		},
		"not-iprequest-ipv4": {
			expectedErrors: 1,
			ipRequest: &allocation.IPRequest{
				ObjectMeta: metav1.ObjectMeta{
					Name: "hello",
				},
			},
		},
		"ip-non-canonical-format": {
			expectedErrors: 1,
			ipRequest: &allocation.IPRequest{
				ObjectMeta: metav1.ObjectMeta{
					Name: "2001:1:2:3:4:0:0:0::2",
				},
			},
		},
	}

	for name, testCase := range testCases {
		t.Run(name, func(t *testing.T) {
			errs := ValidateIPRequest(testCase.ipRequest)
			if len(errs) != testCase.expectedErrors {
				t.Errorf("Expected %d errors, got %d errors: %v", testCase.expectedErrors, len(errs), errs)
			}
		})
	}
}
