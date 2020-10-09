/*
Copyright 2016 The Kubernetes Authors.

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

// Package types defines types used only by volume components
package types

import (
	"errors"

	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/mount-utils"
)

// UniquePodName defines the type to key pods off of
type UniquePodName types.UID

// UniquePVCName defines the type to key pvc off
type UniquePVCName types.UID

// GeneratedOperations contains the operation that is created as well as
// supporting functions required for the operation executor
type GeneratedOperations struct {
	// Name of operation - could be used for resetting shared exponential backoff
	OperationName     string
	OperationFunc     func() (eventErr error, detailedErr error)
	EventRecorderFunc func(*error)
	CompleteFunc      func(*error)
}

// Run executes the operations and its supporting functions
func (o *GeneratedOperations) Run() (eventErr, detailedErr error) {
	if o.CompleteFunc != nil {
		defer o.CompleteFunc(&detailedErr)
	}
	if o.EventRecorderFunc != nil {
		defer o.EventRecorderFunc(&eventErr)
	}
	// Handle panic, if any, from operationFunc()
	defer runtime.RecoverFromPanic(&detailedErr)
	return o.OperationFunc()
}

// FailedPrecondition error indicates CSI operation returned failed precondition
// error
type FailedPrecondition struct {
	msg string
}

func (err *FailedPrecondition) Error() string {
	return err.msg
}

// NewFailedPreconditionError returns a new FailedPrecondition error instance
func NewFailedPreconditionError(msg string) *FailedPrecondition {
	return &FailedPrecondition{msg: msg}
}

// IsFailedPreconditionError checks if given error is of type that indicates
// operation failed with precondition
func IsFailedPreconditionError(err error) bool {
	var failedPreconditionError *FailedPrecondition
	return errors.As(err, &failedPreconditionError)
}

// TransientOperationFailure indicates operation failed with a transient error
// and may fix itself when retried.
type TransientOperationFailure struct {
	msg string
}

func (err *TransientOperationFailure) Error() string {
	return err.msg
}

// NewTransientOperationFailure creates an instance of TransientOperationFailure error
func NewTransientOperationFailure(msg string) *TransientOperationFailure {
	return &TransientOperationFailure{msg: msg}
}

// UncertainProgressError indicates operation failed with a non-final error
// and operation may be in-progress in background.
type UncertainProgressError struct {
	msg string
}

func (err *UncertainProgressError) Error() string {
	return err.msg
}

// NewUncertainProgressError creates an instance of UncertainProgressError type
func NewUncertainProgressError(msg string) *UncertainProgressError {
	return &UncertainProgressError{msg: msg}
}

// IsOperationFinishedError checks if given error is of type that indicates
// operation is finished with a FINAL error.
func IsOperationFinishedError(err error) bool {
	if _, ok := err.(*UncertainProgressError); ok {
		return false
	}
	if _, ok := err.(*TransientOperationFailure); ok {
		return false
	}
	return true
}

// IsFilesystemMismatchError checks if mount failed because requested filesystem
// on PVC and actual filesystem on disk did not match
func IsFilesystemMismatchError(err error) bool {
	mountError := mount.MountError{}
	if errors.As(err, &mountError) && mountError.Type == mount.FilesystemMismatch {
		return true
	}
	return false
}

// IsUncertainProgressError checks if given error is of type that indicates
// operation might be in-progress in background.
func IsUncertainProgressError(err error) bool {
	if _, ok := err.(*UncertainProgressError); ok {
		return true
	}
	return false
}

const (
	// VolumeResizerKey is key that will be used to store resizer used
	// for resizing PVC. The generated key/value pair will be added
	// as a annotation to the PVC.
	VolumeResizerKey = "volume.kubernetes.io/storage-resizer"
)
