/*
Copyright 2015 The Kubernetes Authors.

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

package storage

import (
	"fmt"

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

const (
	ErrCodeKeyNotFound int = iota + 1
	ErrCodeKeyExists
	ErrCodeResourceVersionConflicts
	ErrCodeInvalidObj
	ErrCodeUnreachable
)

var errCodeToMessage = map[int]string{
	ErrCodeKeyNotFound:              "key not found",
	ErrCodeKeyExists:                "key exists",
	ErrCodeResourceVersionConflicts: "resource version conflicts",
	ErrCodeInvalidObj:               "invalid object",
	ErrCodeUnreachable:              "server unreachable",
}

func NewKeyNotFoundError(key string, rv int64) *StorageError {
	return &StorageError{
		Code:            ErrCodeKeyNotFound,
		Key:             key,
		ResourceVersion: rv,
	}
}

func NewKeyExistsError(key string, rv int64) *StorageError {
	return &StorageError{
		Code:            ErrCodeKeyExists,
		Key:             key,
		ResourceVersion: rv,
	}
}

func NewResourceVersionConflictsError(key string, rv int64) *StorageError {
	return &StorageError{
		Code:            ErrCodeResourceVersionConflicts,
		Key:             key,
		ResourceVersion: rv,
	}
}

func NewUnreachableError(key string, rv int64) *StorageError {
	return &StorageError{
		Code:            ErrCodeUnreachable,
		Key:             key,
		ResourceVersion: rv,
	}
}

func NewInvalidObjError(key, msg string) *StorageError {
	return &StorageError{
		Code:               ErrCodeInvalidObj,
		Key:                key,
		AdditionalErrorMsg: msg,
	}
}

type StorageError struct {
	Code               int
	Key                string
	ResourceVersion    int64
	AdditionalErrorMsg string
}

func (e *StorageError) Error() string {
	return fmt.Sprintf("StorageError: %s, Code: %d, Key: %s, ResourceVersion: %d, AdditionalErrorMsg: %s",
		errCodeToMessage[e.Code], e.Code, e.Key, e.ResourceVersion, e.AdditionalErrorMsg)
}

// IsNotFound returns true if and only if err is "key" not found error.
func IsNotFound(err error) bool {
	return isErrCode(err, ErrCodeKeyNotFound)
}

// IsExist returns true if and only if err is "key" already exists error.
func IsExist(err error) bool {
	return isErrCode(err, ErrCodeKeyExists)
}

// IsUnreachable returns true if and only if err indicates the server could not be reached.
func IsUnreachable(err error) bool {
	return isErrCode(err, ErrCodeUnreachable)
}

// IsConflict returns true if and only if err is a write conflict.
func IsConflict(err error) bool {
	return isErrCode(err, ErrCodeResourceVersionConflicts)
}

// IsInvalidObj returns true if and only if err is invalid error
func IsInvalidObj(err error) bool {
	return isErrCode(err, ErrCodeInvalidObj)
}

func isErrCode(err error, code int) bool {
	if err == nil {
		return false
	}
	if e, ok := err.(*StorageError); ok {
		return e.Code == code
	}
	return false
}

// InvalidError is generated when an error caused by invalid API object occurs
// in the storage package.
type InvalidError struct {
	Errs field.ErrorList
}

func (e InvalidError) Error() string {
	return e.Errs.ToAggregate().Error()
}

// IsInvalidError returns true if and only if err is an InvalidError.
func IsInvalidError(err error) bool {
	_, ok := err.(InvalidError)
	return ok
}

func NewInvalidError(errors field.ErrorList) InvalidError {
	return InvalidError{errors}
}

// InternalError is generated when an error occurs in the storage package, i.e.,
// not from the underlying storage backend (e.g., etcd).
type InternalError struct {
	Reason string
}

func (e InternalError) Error() string {
	return e.Reason
}

// IsInternalError returns true if and only if err is an InternalError.
func IsInternalError(err error) bool {
	_, ok := err.(InternalError)
	return ok
}

func NewInternalError(reason string) InternalError {
	return InternalError{reason}
}

func NewInternalErrorf(format string, a ...interface{}) InternalError {
	return InternalError{fmt.Sprintf(format, a...)}
}

var tooLargeResourceVersionCauseMsg = "Too large resource version"

// NewTooLargeResourceVersionError returns a timeout error with the given retrySeconds for a request for
// a minimum resource version that is larger than the largest currently available resource version for a requested resource.
func NewTooLargeResourceVersionError(minimumResourceVersion, currentRevision uint64, retrySeconds int) error {
	err := errors.NewTimeoutError(fmt.Sprintf("Too large resource version: %d, current: %d", minimumResourceVersion, currentRevision), retrySeconds)
	err.ErrStatus.Details.Causes = []metav1.StatusCause{
		{
			Type:    metav1.CauseTypeResourceVersionTooLarge,
			Message: tooLargeResourceVersionCauseMsg,
		},
	}
	return err
}

// IsTooLargeResourceVersion returns true if the error is a TooLargeResourceVersion error.
func IsTooLargeResourceVersion(err error) bool {
	if !errors.IsTimeout(err) {
		return false
	}
	return errors.HasStatusCause(err, metav1.CauseTypeResourceVersionTooLarge)
}
