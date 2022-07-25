/*
Copyright 2020 The Kubernetes Authors.

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

package types

import (
	"fmt"
	"testing"

	"k8s.io/mount-utils"
)

func TestErrorTypes(t *testing.T) {
	tests := []struct {
		name           string
		realError      error
		errorCheckFunc func(error) bool
		expectError    bool
	}{
		{
			"when mount error has File system mismatch errors",
			mount.NewMountError(mount.FilesystemMismatch, "filesystem mismatch"),
			IsFilesystemMismatchError,
			true,
		},
		{
			"when mount error has other error",
			mount.NewMountError(mount.FormatFailed, "filesystem mismatch"),
			IsFilesystemMismatchError,
			false,
		},
		{
			"when mount error wraps filesystem mismatch error",
			fmt.Errorf("mount failed %w", mount.NewMountError(mount.FilesystemMismatch, "filesystem mismatch")),
			IsFilesystemMismatchError,
			true,
		},
		{
			"when error has no failedPrecondition error",
			fmt.Errorf("some other error"),
			IsFailedPreconditionError,
			false,
		},
		{
			"when error has failedPrecondition error",
			NewFailedPreconditionError("volume-in-use"),
			IsFailedPreconditionError,
			true,
		},
		{
			"when error wraps failedPrecondition error",
			fmt.Errorf("volume readonly %w", NewFailedPreconditionError("volume-in-use-error")),
			IsFailedPreconditionError,
			true,
		},
	}

	for _, test := range tests {
		ok := test.errorCheckFunc(test.realError)
		if ok != test.expectError {
			t.Errorf("for %s: expected error to be %v but got %v", test.name, test.expectError, ok)
		}
	}
}
