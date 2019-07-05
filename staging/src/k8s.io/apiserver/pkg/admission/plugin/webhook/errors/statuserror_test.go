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

package errors

import (
	"fmt"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestToStatusErr(t *testing.T) {
	hookName := "foo"
	deniedBy := fmt.Sprintf("admission webhook %q denied the request", hookName)
	tests := []struct {
		name           string
		result         *metav1.Status
		expectedError  string
		expectedCode   int32
		expectedStatus string
	}{
		{
			"nil result",
			nil,
			deniedBy + " without explanation",
			400,
			metav1.StatusFailure,
		},
		{
			"only message",
			&metav1.Status{
				Message: "you shall not pass",
			},
			deniedBy + ": you shall not pass",
			400,
			metav1.StatusFailure,
		},
		{
			"only reason",
			&metav1.Status{
				Reason: metav1.StatusReasonForbidden,
			},
			deniedBy + ": Forbidden",
			400,
			metav1.StatusFailure,
		},
		{
			"message and reason",
			&metav1.Status{
				Message: "you shall not pass",
				Reason:  metav1.StatusReasonForbidden,
			},
			deniedBy + ": you shall not pass",
			400,
			metav1.StatusFailure,
		},
		{
			"no message, no reason",
			&metav1.Status{},
			deniedBy + " without explanation",
			400,
			metav1.StatusFailure,
		},
		{
			"custom 4xx status code",
			&metav1.Status{Code: 401},
			deniedBy + " without explanation",
			401,
			metav1.StatusFailure,
		},
		{
			"custom 5xx status code",
			&metav1.Status{Code: 500},
			deniedBy + " without explanation",
			500,
			metav1.StatusFailure,
		},
		{
			"200 status code",
			&metav1.Status{Code: 200},
			deniedBy + " without explanation",
			400,
			metav1.StatusFailure,
		},
		{
			"300 status code",
			&metav1.Status{Code: 300},
			deniedBy + " without explanation",
			400,
			metav1.StatusFailure,
		},
		{
			"399 status code",
			&metav1.Status{Code: 399},
			deniedBy + " without explanation",
			400,
			metav1.StatusFailure,
		},
		{
			"missing status",
			&metav1.Status{},
			deniedBy + " without explanation",
			400,
			metav1.StatusFailure,
		},
		{
			"success status overridden",
			&metav1.Status{Status: metav1.StatusSuccess},
			deniedBy + " without explanation",
			400,
			metav1.StatusFailure,
		},
		{
			"failure status preserved",
			&metav1.Status{Status: metav1.StatusFailure},
			deniedBy + " without explanation",
			400,
			metav1.StatusFailure,
		},
		{
			"custom status preserved",
			&metav1.Status{Status: "custom"},
			deniedBy + " without explanation",
			400,
			"custom",
		},
	}
	for _, test := range tests {
		err := ToStatusErr(hookName, test.result)
		if err == nil || err.Error() != test.expectedError {
			t.Errorf("%s: expected an error saying %q, but got %v", test.name, test.expectedError, err)
		}
		if err.ErrStatus.Code != test.expectedCode {
			t.Errorf("%s: expected code %d, got %d", test.name, test.expectedCode, err.ErrStatus.Code)
		}
		if err.ErrStatus.Status != test.expectedStatus {
			t.Errorf("%s: expected code %q, got %q", test.name, test.expectedStatus, err.ErrStatus.Status)
		}
	}
}
