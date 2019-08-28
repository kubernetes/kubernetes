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

package v1alpha1

import (
	"errors"
	"testing"
)

func TestStatus(t *testing.T) {
	tests := []struct {
		status            *Status
		expectedCode      Code
		expectedMessage   string
		expectedIsSuccess bool
		expectedAsError   error
	}{
		{
			status:            NewStatus(Success, ""),
			expectedCode:      Success,
			expectedMessage:   "",
			expectedIsSuccess: true,
			expectedAsError:   nil,
		},
		{
			status:            NewStatus(Error, "unknown error"),
			expectedCode:      Error,
			expectedMessage:   "unknown error",
			expectedIsSuccess: false,
			expectedAsError:   errors.New("unknown error"),
		},
		{
			status:            nil,
			expectedCode:      Success,
			expectedMessage:   "",
			expectedIsSuccess: true,
			expectedAsError:   nil,
		},
	}

	for i, test := range tests {
		if test.status.Code() != test.expectedCode {
			t.Errorf("test #%v, expect status.Code() returns %v, but %v", i, test.expectedCode, test.status.Code())
		}

		if test.status.Message() != test.expectedMessage {
			t.Errorf("test #%v, expect status.Message() returns %v, but %v", i, test.expectedMessage, test.status.Message())
		}

		if test.status.IsSuccess() != test.expectedIsSuccess {
			t.Errorf("test #%v, expect status.IsSuccess() returns %v, but %v", i, test.expectedIsSuccess, test.status.IsSuccess())
		}

		if test.status.AsError() == test.expectedAsError {
			continue
		}

		if test.status.AsError().Error() != test.expectedAsError.Error() {
			t.Errorf("test #%v, expect status.AsError() returns %v, but %v", i, test.expectedAsError, test.status.AsError())
		}
	}
}
