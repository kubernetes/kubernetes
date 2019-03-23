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
		name          string
		result        *metav1.Status
		expectedError string
	}{
		{
			"nil result",
			nil,
			deniedBy + " without explanation",
		},
		{
			"only message",
			&metav1.Status{
				Message: "you shall not pass",
			},
			deniedBy + ": you shall not pass",
		},
		{
			"only reason",
			&metav1.Status{
				Reason: metav1.StatusReasonForbidden,
			},
			deniedBy + ": Forbidden",
		},
		{
			"message and reason",
			&metav1.Status{
				Message: "you shall not pass",
				Reason:  metav1.StatusReasonForbidden,
			},
			deniedBy + ": you shall not pass",
		},
		{
			"no message, no reason",
			&metav1.Status{},
			deniedBy + " without explanation",
		},
	}
	for _, test := range tests {
		err := ToStatusErr(hookName, test.result)
		if err == nil || err.Error() != test.expectedError {
			t.Errorf("%s: expected an error saying %q, but got %v", test.name, test.expectedError, err)
		}
	}
}
