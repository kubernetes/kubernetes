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

package framework

import (
	"errors"
	"testing"
)

func TestStatus(t *testing.T) {
	tests := []struct {
		name              string
		status            *Status
		expectedCode      Code
		expectedMessage   string
		expectedIsSuccess bool
		expectedAsError   error
	}{
		{
			name:              "success status",
			status:            NewStatus(Success, ""),
			expectedCode:      Success,
			expectedMessage:   "",
			expectedIsSuccess: true,
			expectedAsError:   nil,
		},
		{
			name:              "error status",
			status:            NewStatus(Error, "unknown error"),
			expectedCode:      Error,
			expectedMessage:   "unknown error",
			expectedIsSuccess: false,
			expectedAsError:   errors.New("unknown error"),
		},
		{
			name:              "nil status",
			status:            nil,
			expectedCode:      Success,
			expectedMessage:   "",
			expectedIsSuccess: true,
			expectedAsError:   nil,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if test.status.Code() != test.expectedCode {
				t.Errorf("expect status.Code() returns %v, but %v", test.expectedCode, test.status.Code())
			}

			if test.status.Message() != test.expectedMessage {
				t.Errorf("expect status.Message() returns %v, but %v", test.expectedMessage, test.status.Message())
			}

			if test.status.IsSuccess() != test.expectedIsSuccess {
				t.Errorf("expect status.IsSuccess() returns %v, but %v", test.expectedIsSuccess, test.status.IsSuccess())
			}

			if test.status.AsError() == test.expectedAsError {
				return
			}

			if test.status.AsError().Error() != test.expectedAsError.Error() {
				t.Errorf("expect status.AsError() returns %v, but %v", test.expectedAsError, test.status.AsError())
			}
		})
	}
}

// The String() method relies on the value and order of the status codes to function properly.
func TestStatusCodes(t *testing.T) {
	assertStatusCode(t, Success, 0)
	assertStatusCode(t, Error, 1)
	assertStatusCode(t, Unschedulable, 2)
	assertStatusCode(t, UnschedulableAndUnresolvable, 3)
	assertStatusCode(t, Wait, 4)
	assertStatusCode(t, Skip, 5)
}

func assertStatusCode(t *testing.T, code Code, value int) {
	if int(code) != value {
		t.Errorf("Status code %q should have a value of %v but got %v", code.String(), value, int(code))
	}
}

func TestPluginToStatusMerge(t *testing.T) {
	tests := []struct {
		name      string
		statusMap PluginToStatus
		wantCode  Code
	}{
		{
			name:      "merge Error and Unschedulable statuses",
			statusMap: PluginToStatus{"p1": NewStatus(Error), "p2": NewStatus(Unschedulable)},
			wantCode:  Error,
		},
		{
			name:      "merge Success and Unschedulable statuses",
			statusMap: PluginToStatus{"p1": NewStatus(Success), "p2": NewStatus(Unschedulable)},
			wantCode:  Unschedulable,
		},
		{
			name:      "merge Success, UnschedulableAndUnresolvable and Unschedulable statuses",
			statusMap: PluginToStatus{"p1": NewStatus(Success), "p2": NewStatus(UnschedulableAndUnresolvable), "p3": NewStatus(Unschedulable)},
			wantCode:  UnschedulableAndUnresolvable,
		},
		{
			name:     "merge nil status",
			wantCode: Success,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			gotStatus := test.statusMap.Merge()
			if test.wantCode != gotStatus.Code() {
				t.Errorf("wantCode %v, gotCode %v", test.wantCode, gotStatus.Code())
			}
		})
	}
}
