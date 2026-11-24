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

	v1 "k8s.io/api/core/v1"
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
)

const (
	ErrorReasonUnexpected         = "UnexpectedAdmissionError"
	ErrorReasonEmptyPodSharedPool = "EmptyPodSharedPoolError"
)

type Error interface {
	Error() string
	Type() string
	RejectionType() lifecycle.RejectionType
}

type unexpectedAdmissionError struct{ Err error }

var _ Error = (*unexpectedAdmissionError)(nil)

func (e *unexpectedAdmissionError) Error() string {
	return fmt.Sprintf("Allocate failed due to %v, which is unexpected", e.Err)
}

func (e *unexpectedAdmissionError) Type() string {
	return ErrorReasonUnexpected
}

func (e *unexpectedAdmissionError) RejectionType() lifecycle.RejectionType {
	return lifecycle.RejectionTransient
}

// EmptyPodSharedPoolError represents an error due to a pod spec being invalid
// when the Topology manager's scope is set to pod, because the pod spec
// produces an empty pod shared pool and there are containers that require it.
type EmptyPodSharedPoolError struct {
	message string
}

var _ Error = (*EmptyPodSharedPoolError)(nil)

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

func (e *EmptyPodSharedPoolError) RejectionType() lifecycle.RejectionType {
	return lifecycle.RejectionInfeasible
}

// OutOfResourceError represents a failure to allocate exclusive resources, such as CPUs or
// NUMA-aligned memory, because not enough suitable resources are free on the node.
type OutOfResourceError struct {
	Resource v1.ResourceName
	Err      error
}

var _ Error = (*OutOfResourceError)(nil)

// NewOutOfResourceError returns a new OutOfResourceError.
func NewOutOfResourceError(resource v1.ResourceName, err error) *OutOfResourceError {
	return &OutOfResourceError{Resource: resource, Err: err}
}

func (e *OutOfResourceError) Error() string {
	return e.Err.Error()
}

func (e *OutOfResourceError) Unwrap() error {
	return e.Err
}

// Type returns the reason in the same form as the OutOf<resource> reasons of the predicate
// admit handler, such as OutOfcpu.
func (e *OutOfResourceError) Type() string {
	return "OutOf" + string(e.Resource)
}

// RejectionType returns RejectionBlocked because resources may become available once other pods
// release them.
// TODO: Have the resource managers return an infeasible error instead when the request can't be
// satisfied even on an otherwise empty node.
func (e *OutOfResourceError) RejectionType() lifecycle.RejectionType {
	return lifecycle.RejectionBlocked
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
		Message:       admissionErr.Error(),
		Reason:        admissionErr.Type(),
		Admit:         false,
		RejectionType: admissionErr.RejectionType(),
	}
}
