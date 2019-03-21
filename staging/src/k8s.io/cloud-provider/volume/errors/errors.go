/*
Copyright 2017 The Kubernetes Authors.

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

package errors

import (
	k8stypes "k8s.io/apimachinery/pkg/types"
)

// NewDeletedVolumeInUseError returns a new instance of DeletedVolumeInUseError
// error.
func NewDeletedVolumeInUseError(message string) error {
	return deletedVolumeInUseError(message)
}

type deletedVolumeInUseError string

var _ error = deletedVolumeInUseError("")

// IsDeletedVolumeInUse returns true if an error returned from Delete() is
// deletedVolumeInUseError
func IsDeletedVolumeInUse(err error) bool {
	switch err.(type) {
	case deletedVolumeInUseError:
		return true
	default:
		return false
	}
}

func (err deletedVolumeInUseError) Error() string {
	return string(err)
}

// DanglingAttachError indicates volume is attached to a different node
// than we expected.
type DanglingAttachError struct {
	msg         string
	CurrentNode k8stypes.NodeName
	DevicePath  string
}

func (err *DanglingAttachError) Error() string {
	return err.msg
}

// NewDanglingError create a new dangling error
func NewDanglingError(msg string, node k8stypes.NodeName, devicePath string) error {
	return &DanglingAttachError{
		msg:         msg,
		CurrentNode: node,
		DevicePath:  devicePath,
	}
}
