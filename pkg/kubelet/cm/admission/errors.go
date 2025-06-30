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

	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
)

const (
	ErrorReasonUnexpected = "UnexpectedAdmissionError"

	// Explicit reason when CPU/Memory manager's policy is incompatible with pod level resources.
	PodLevelResourcesIncompatible = "PodLevelResourcesIncompatible"

	// Warnings for pod level resources when manager's policy incompatibility.
	CPUManagerPodLevelResourcesError    = "CPUManagerPodLevelResourcesError"
	MemoryManagerPodLevelResourcesError = "MemoryManagerPodLevelResourcesError"
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

	var errs []error
	// To support multiple pod-level resource errors, we need to check if the error
	// is an aggregate error.
	var agg utilerrors.Aggregate
	if errors.As(err, &agg) {
		errs = agg.Errors()
	} else {
		errs = []error{err}
	}

	var podLevelWarnings []error
	var otherErrs []error
	for _, e := range errs {
		var admissionErr Error
		if errors.As(e, &admissionErr) && (admissionErr.Type() == CPUManagerPodLevelResourcesError || admissionErr.Type() == MemoryManagerPodLevelResourcesError) {
			podLevelWarnings = append(podLevelWarnings, e)
		} else {
			otherErrs = append(otherErrs, e)
		}
	}

	// If all errors are pod-level resource errors, we should treat them as warnings
	// and not block pod admission.
	if len(otherErrs) == 0 && len(podLevelWarnings) > 0 {
		return lifecycle.PodAdmitResult{
			Admit:   true,
			Reason:  PodLevelResourcesIncompatible,
			Message: "",
			Errors:  podLevelWarnings,
		}
	}

	if len(otherErrs) == 0 {
		// This should not happen if err != nil, but as a safeguard.
		return lifecycle.PodAdmitResult{Admit: true}
	}

	// At this point, we have at least one error that requires pod rejection.
	firstErr := otherErrs[0]
	var admissionErr Error
	if !errors.As(firstErr, &admissionErr) {
		admissionErr = &unexpectedAdmissionError{firstErr}
	}

	return lifecycle.PodAdmitResult{
		Admit:   false,
		Message: admissionErr.Error(),
		Reason:  admissionErr.Type(),
	}
}
