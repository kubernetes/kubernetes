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
	ErrorReasonUnexpected = "UnexpectedAdmissionError"
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

type retryableError interface {
	Error() string
	Type() string
	IsRetryable() bool
}

type retryableAdmissionError struct{ Err error }

var _ Error = (*retryableAdmissionError)(nil)

func (e *retryableAdmissionError) Error() string {
	return e.Err.Error()
}

func (e *retryableAdmissionError) Type() string {
	return ErrorReasonUnexpected
}

func (e *retryableAdmissionError) IsRetryable() bool {
	return true
}

// NewRetryableAdmissionError indicates that an admission error may
// be retried.
// At the time of implementation, a single retry will be performed with a 5s delay.
//
// note: Initially this is only used by the devicemanager to account
//
//	for node restarts and races with plugins coming up.
//	It is an unclean hack to reduce the scope of the topologymanager refactor.
func NewRetryableAdmissionError(err error) Error {
	if err == nil {
		return nil
	}
	return &retryableAdmissionError{Err: err}
}

func GetPodAdmitResult(err error) (lifecycle.PodAdmitResult, error) {
	if err == nil {
		return lifecycle.PodAdmitResult{Admit: true}, nil
	}

	var retryableErr retryableError
	if errors.As(err, &retryableErr) && retryableErr.IsRetryable() {
		admissionErr := &unexpectedAdmissionError{err}
		return lifecycle.PodAdmitResult{
			Message: admissionErr.Error(),
			Reason:  admissionErr.Type(),
			Admit:   false,
		}, retryableErr
	}

	var admissionErr Error
	if !errors.As(err, &admissionErr) {
		admissionErr = &unexpectedAdmissionError{err}
	}

	return lifecycle.PodAdmitResult{
		Message: admissionErr.Error(),
		Reason:  admissionErr.Type(),
		Admit:   false,
	}, nil
}
