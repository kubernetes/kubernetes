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

	v1 "k8s.io/api/core/v1"
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
)

type TestAdmissionError struct {
	message       string
	reason        string
	rejectionType lifecycle.RejectionType
}

func (e *TestAdmissionError) Error() string {
	return e.message
}

func (e *TestAdmissionError) Type() string {
	return e.reason
}

func (e *TestAdmissionError) RejectionType() lifecycle.RejectionType {
	return e.rejectionType
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
				message: "Is an AdmissionError error",
				reason:  "TestAdmissionError",
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

func TestGetPodAdmitResultRejectionType(t *testing.T) {
	testCases := []struct {
		name                  string
		err                   error
		expectedRejectionType lifecycle.RejectionType
	}{{
		name:                  "untyped error",
		err:                   errors.New("unexpected"),
		expectedRejectionType: lifecycle.RejectionTransient,
	}, {
		name:                  "custom admission error",
		err:                   &TestAdmissionError{message: "custom", reason: "Custom", rejectionType: lifecycle.RejectionInfeasible},
		expectedRejectionType: lifecycle.RejectionInfeasible,
	}, {
		name:                  "EmptyPodSharedPoolError",
		err:                   NewEmptyPodSharedPoolError(errors.New("empty shared pool")),
		expectedRejectionType: lifecycle.RejectionInfeasible,
	}, {
		name:                  "OutOfResourceError for cpu",
		err:                   NewOutOfResourceError(v1.ResourceCPU, errors.New("not enough cpus")),
		expectedRejectionType: lifecycle.RejectionBlocked,
	}, {
		name:                  "OutOfResourceError for memory",
		err:                   NewOutOfResourceError(v1.ResourceMemory, errors.New("not enough memory")),
		expectedRejectionType: lifecycle.RejectionBlocked,
	}, {
		name:                  "wrapped OutOfResourceError",
		err:                   fmt.Errorf("wrapped: %w", NewOutOfResourceError(v1.ResourceCPU, errors.New("not enough cpus"))),
		expectedRejectionType: lifecycle.RejectionBlocked,
	}}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			h := GetPodAdmitResult(tc.err)
			if h.Admit {
				t.Fatalf("expected PodAdmitResult.Admit = false")
			}
			if h.RejectionType != tc.expectedRejectionType {
				t.Errorf("expected PodAdmitResult.RejectionType = %v, got %v", tc.expectedRejectionType, h.RejectionType)
			}
		})
	}
}
