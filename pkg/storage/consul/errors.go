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

package consul

import (
	"fmt"
	"net/url"

	"k8s.io/kubernetes/pkg/storage"
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

func isURLError(err error) bool {
	if err != nil {
		if _, ok := err.(*url.Error); ok {
			return true
		}
	}
	return false
}

func isNotFoundError(err error) bool {
	if err != nil {
		if consulError, ok := err.(ConsulError); ok {
			return consulError.Code == ErrCodeKeyNotFound
		}
	}
	return false
}

func isInvalidObjectError(err error) bool {
	if err != nil {
		if consulError, ok := err.(ConsulError); ok {
			return consulError.Code == ErrCodeInvalidObj
		}
	}
	return false
}

func isExists(err error) bool {
	if err != nil {
		if consulError, ok := err.(ConsulError); ok {
			return consulError.Code == ErrCodeKeyExists
		}
	}
	return false
}

func isConflict(err error) bool {
	if err != nil {
		if consulError, ok := err.(ConsulError); ok {
			return consulError.Code == ErrCodeResourceVersionConflicts
		}
	}
	return false
}

func NewInvalidObjectError(msg string) ConsulError {
	return ConsulError{
		Code:    ErrCodeInvalidObj,
		Message: msg,
	}
}

func NewNotFoundError() ConsulError {
	return ConsulError{
		Code: storage.ErrCodeKeyNotFound,
	}
}

func NewExistsError(msg string) ConsulError {
	return ConsulError{
		Code:    storage.ErrCodeKeyExists,
		Message: msg,
	}
}

func NewConflictError(msg string) ConsulError {
	return ConsulError{
		Code:    storage.ErrCodeResourceVersionConflicts,
		Message: msg,
	}
}

type ConsulError struct {
	Code    int
	Message string
}

func (e ConsulError) Error() string {
	return fmt.Sprintf("StorageError: %s, Code: %d, Key: %s, ResourceVersion: %d, AdditionalErrorMsg: %s",
		errCodeToMessage[e.Code], e.Code, "", 0, e.Message)
}
