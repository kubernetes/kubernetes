// Copyright 2016 The etcd Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// +build darwin

package fileutil

import (
	"os"
	"syscall"
	"unsafe"
)

func preallocExtend(f *os.File, sizeInBytes int64) error {
	if err := preallocFixed(f, sizeInBytes); err != nil {
		return err
	}
	return preallocExtendTrunc(f, sizeInBytes)
}

func preallocFixed(f *os.File, sizeInBytes int64) error {
	fstore := &syscall.Fstore_t{
		Flags:   syscall.F_ALLOCATEALL,
		Posmode: syscall.F_PEOFPOSMODE,
		Length:  sizeInBytes}
	p := unsafe.Pointer(fstore)
	_, _, errno := syscall.Syscall(syscall.SYS_FCNTL, f.Fd(), uintptr(syscall.F_PREALLOCATE), uintptr(p))
	if errno == 0 || errno == syscall.ENOTSUP {
		return nil
	}
	return errno
}
