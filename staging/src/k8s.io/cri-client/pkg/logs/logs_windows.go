//go:build windows
// +build windows

/*
Copyright 2024 The Kubernetes Authors.

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

package logs

import (
	"os"
	"syscall"
)

// Based on Windows implementation of Windows' syscall.Open
// https://cs.opensource.google/go/go/+/refs/tags/go1.22.2:src/syscall/syscall_windows.go;l=342
// In addition to syscall.Open, this function also adds the syscall.FILE_SHARE_DELETE flag to sharemode,
// which will allow us to read from the file without blocking the file from being deleted or renamed.
// This is essential for Log Rotation which is done by renaming the open file. Without this, the file rename would fail.
func openFileShareDelete(path string) (*os.File, error) {
	pathp, err := syscall.UTF16PtrFromString(path)
	if err != nil {
		return nil, err
	}

	var access uint32 = syscall.GENERIC_READ
	var sharemode uint32 = syscall.FILE_SHARE_READ | syscall.FILE_SHARE_WRITE | syscall.FILE_SHARE_DELETE
	var createmode uint32 = syscall.OPEN_EXISTING
	var attrs uint32 = syscall.FILE_ATTRIBUTE_NORMAL

	handle, err := syscall.CreateFile(pathp, access, sharemode, nil, createmode, attrs, 0)
	if err != nil {
		return nil, err
	}

	return os.NewFile(uintptr(handle), path), nil
}
