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

	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
)

const (
	ErrorReasonUnexpected         = "UnexpectedAdmissionError"
	ErrorReasonEmptyPodSharedPool = "EmptyPodSharedPoolError"
	ErrorReasonDeviceNotReady     = "DeviceNotReady"
)

type Error interface {
	Error() string
	Type() string
}

type unexpectedAdmissionError struct{ Err error }

var _ Error = (*unexpectedAdmissionError)(nil)

func (e *unexpectedAdmissionError) Error() string {
	return fmt.Sprintf("Allocate failed due to %v, which is unexpected", e.Err)
}

func (e *unexpectedAdmissionError) Type() string {
	return ErrorReasonUnexpected
}

// EmptyPodSharedPoolError represents an error due to a pod spec being invalid
// when the Topology manager's scope is set to pod, because the pod spec
// produces an empty pod shared pool and there are containers that require it.
type EmptyPodSharedPoolError struct {
	message string
}

// NewEmptyPodSharedPoolError returns a new EmptyPodSharedPoolError.
func NewEmptyPodSharedPoolError(err error) *EmptyPodSharedPoolError {
	return &EmptyPodSharedPoolError{message: err.Error()}
}

func (e *EmptyPodSharedPoolError) Error() string {
	return e.message
}

func (e *EmptyPodSharedPoolError) Type() string {
	return ErrorReasonEmptyPodSharedPool
}

// DeviceNotReadyError represents an admission failure that should be deferred
// (retried later) rather than permanently rejected. It is used when a pod
// requests a device plugin resource whose device plugin has not yet registered
// with the kubelet. The device plugin may register shortly, so the pod is kept
// Pending and admission is retried instead of failing the pod immediately.
type DeviceNotReadyError struct {
	Err error
}

var _ Error = (*DeviceNotReadyError)(nil)

// NewDeviceNotReadyError returns a new DeviceNotReadyError wrapping err.
func NewDeviceNotReadyError(err error) *DeviceNotReadyError {
	return &DeviceNotReadyError{Err: err}
}

func (e *DeviceNotReadyError) Error() string {
	return e.Err.Error()
}

func (e *DeviceNotReadyError) Type() string {
	return ErrorReasonDeviceNotReady
}

// Unwrap allows errors.As/errors.Is to traverse the wrapped error.
func (e *DeviceNotReadyError) Unwrap() error {
	return e.Err
}

func GetPodAdmitResult(err error) lifecycle.PodAdmitResult {
	if err == nil {
		return lifecycle.PodAdmitResult{Admit: true}
	}

	var admissionErr Error
	if !errors.As(err, &admissionErr) {
		admissionErr = &unexpectedAdmissionError{err}
	}

	// Device-not-ready errors are deferrable: the device plugin may register
	// shortly, so the pod should be retried rather than permanently rejected.
	var deviceNotReadyErr *DeviceNotReadyError
	deferAdmission := errors.As(err, &deviceNotReadyErr)

	return lifecycle.PodAdmitResult{
		Message: admissionErr.Error(),
		Reason:  admissionErr.Type(),
		Admit:   false,
		Defer:   deferAdmission,
	}
}
