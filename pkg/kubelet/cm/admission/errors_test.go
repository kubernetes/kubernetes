/*
Copyright 2021 The Kubernetes Authors.

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

package admission

import (
	"errors"
	"fmt"
	"testing"
)

type TestAdmissionError struct {
	message string
	reason  string
}

func (e *TestAdmissionError) Error() string {
	return e.message
}

func (e *TestAdmissionError) Type() string {
	return e.reason
}

func TestAdmissionErrors(t *testing.T) {
	testCases := []struct {
		Error                  error
		expectedAdmissionError bool
	}{
		{
			nil,
			false,
		},
		{
			errors.New("Not an AdmissionError error"),
			false,
		},
		{
			&TestAdmissionError{
				"Is an AdmissionError error",
				"TestAdmissionError",
			},
			true,
		},
	}

	for _, tc := range testCases {
		h := GetPodAdmitResult(tc.Error)
		if tc.Error == nil {
			if !h.Admit {
				t.Errorf("expected PodAdmitResult.Admit = true")
			}
			continue
		}

		if h.Admit {
			t.Errorf("expected PodAdmitResult.Admit = false")
		}

		if tc.expectedAdmissionError {
			err, ok := tc.Error.(*TestAdmissionError)
			if !ok {
				t.Errorf("expected TestAdmissionError")
			}
			if h.Reason != err.reason {
				t.Errorf("expected PodAdmitResult.Reason = %v, got %v", err.reason, h.Reason)
			}
			continue
		}

		if h.Reason != ErrorReasonUnexpected {
			t.Errorf("expected PodAdmitResult.Reason = %v, got %v", ErrorReasonUnexpected, h.Reason)
		}
	}
}

func TestDeviceNotReadyError(t *testing.T) {
	innerErr := fmt.Errorf("cannot allocate unregistered device example.com/gpu")
	devErr := NewDeviceNotReadyError(innerErr)

	// It must satisfy the Error interface with the expected type.
	if devErr.Type() != ErrorReasonDeviceNotReady {
		t.Errorf("expected Type() = %q, got %q", ErrorReasonDeviceNotReady, devErr.Type())
	}
	if devErr.Error() != innerErr.Error() {
		t.Errorf("expected Error() = %q, got %q", innerErr.Error(), devErr.Error())
	}

	// errors.As should find the DeviceNotReadyError even when wrapped.
	wrapped := fmt.Errorf("admission failed: %w", devErr)
	var asErr *DeviceNotReadyError
	if !errors.As(wrapped, &asErr) {
		t.Fatalf("expected errors.As to find *DeviceNotReadyError in wrapped error")
	}

	// GetPodAdmitResult must set Defer: true for a device-not-ready error.
	result := GetPodAdmitResult(devErr)
	if result.Admit {
		t.Errorf("expected PodAdmitResult.Admit = false")
	}
	if !result.Defer {
		t.Errorf("expected PodAdmitResult.Defer = true for DeviceNotReadyError")
	}
	if result.Reason != ErrorReasonDeviceNotReady {
		t.Errorf("expected PodAdmitResult.Reason = %q, got %q", ErrorReasonDeviceNotReady, result.Reason)
	}

	// A non-deferrable error must not set Defer.
	otherResult := GetPodAdmitResult(errors.New("some other error"))
	if otherResult.Defer {
		t.Errorf("expected PodAdmitResult.Defer = false for non-device error")
	}
}
