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

func GetPodAdmitResult(err error) lifecycle.PodAdmitResult {
	if err == nil {
		return lifecycle.PodAdmitResult{Admit: true}
	}

	var admissionErr Error
	if !errors.As(err, &admissionErr) {
		admissionErr = &unexpectedAdmissionError{err}
	}

	return lifecycle.PodAdmitResult{
		Message: admissionErr.Error(),
		Reason:  admissionErr.Type(),
		Admit:   false,
	}
}
