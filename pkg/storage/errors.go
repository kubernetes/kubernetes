/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

import "fmt"

const (
	ErrCodeKeyNotFound int = iota + 1
	ErrCodeKeyExists
	ErrCodeResourceVersionConflicts
	ErrCodeUnreachable
)

var errCodeToMessage = map[int]string{
	ErrCodeKeyNotFound:              "key not found",
	ErrCodeKeyExists:                "key exists",
	ErrCodeResourceVersionConflicts: "resource version conflicts",
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

type StorageError struct {
	Code            int
	Key             string
	ResourceVersion int64
}

func (e *StorageError) Error() string {
	return fmt.Sprintf("StorageError: %s, Code: %d, Key: %s, ResourceVersion: %d",
		errCodeToMessage[e.Code], e.Code, e.Key, e.ResourceVersion)
}

// IsNotFound returns true if and only if err is "key" not found error.
func IsNotFound(err error) bool {
	return isErrCode(err, ErrCodeKeyNotFound)
}

// IsNodeExist returns true if and only if err is an node already exist error.
func IsNodeExist(err error) bool {
	return isErrCode(err, ErrCodeKeyExists)
}

// IsUnreachable returns true if and only if err indicates the server could not be reached.
func IsUnreachable(err error) bool {
	return isErrCode(err, ErrCodeUnreachable)
}

// IsTestFailed returns true if and only if err is a write conflict.
func IsTestFailed(err error) bool {
	return isErrCode(err, ErrCodeResourceVersionConflicts)
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
