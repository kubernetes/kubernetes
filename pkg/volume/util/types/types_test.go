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

	"k8s.io/utils/mount"
)

func TestIsFilesystemMismatchError(t *testing.T) {
	tests := []struct {
		mountError  error
		expectError bool
	}{
		{
			mount.NewMountError(mount.FilesystemMismatch, "filesystem mismatch"),
			true,
		},
		{
			mount.NewMountError(mount.FormatFailed, "filesystem mismatch"),
			false,
		},
		{
			fmt.Errorf("mount failed %w", mount.NewMountError(mount.FilesystemMismatch, "filesystem mismatch")),
			true,
		},
	}
	for _, test := range tests {
		ok := IsFilesystemMismatchError(test.mountError)
		if ok != test.expectError {
			t.Errorf("expected filesystem mismatch to be %v but got %v", test.expectError, ok)
		}
	}
}
