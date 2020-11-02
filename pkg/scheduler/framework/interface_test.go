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
		statusMap PluginToStatus
		wantCode  Code
	}{
		{
			statusMap: PluginToStatus{"p1": NewStatus(Error), "p2": NewStatus(Unschedulable)},
			wantCode:  Error,
		},
		{
			statusMap: PluginToStatus{"p1": NewStatus(Success), "p2": NewStatus(Unschedulable)},
			wantCode:  Unschedulable,
		},
		{
			statusMap: PluginToStatus{"p1": NewStatus(Success), "p2": NewStatus(UnschedulableAndUnresolvable), "p3": NewStatus(Unschedulable)},
			wantCode:  UnschedulableAndUnresolvable,
		},
		{
			wantCode: Success,
		},
	}

	for i, test := range tests {
		gotStatus := test.statusMap.Merge()
		if test.wantCode != gotStatus.Code() {
			t.Errorf("test #%v, wantCode %v, gotCode %v", i, test.wantCode, gotStatus.Code())
		}
	}
}
