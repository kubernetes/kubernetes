// Copyright 2015 The go4 Authors
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

// +build amd64,solaris

package osutil

import (
	"os"
	"syscall"
	"unsafe"
)

//go:cgo_import_dynamic libc_getexecname getexecname "libc.so"
//go:linkname libc_getexecname libc_getexecname

var libc_getexecname uintptr

func getexecname() (path unsafe.Pointer, err error) {
	r0, _, e1 := syscall.Syscall6(uintptr(unsafe.Pointer(&libc_getexecname)), 0, 0, 0, 0, 0, 0)
	path = unsafe.Pointer(r0)
	if e1 != 0 {
		err = syscall.Errno(e1)
	}
	return
}

func syscallGetexecname() (path string, err error) {
	ptr, err := getexecname()
	if err != nil {
		return "", err
	}
	bytes := (*[1 << 29]byte)(ptr)[:]
	for i, b := range bytes {
		if b == 0 {
			return string(bytes[:i]), nil
		}
	}
	panic("unreachable")
}

var initCwd, initCwdErr = os.Getwd()

func executable() (string, error) {
	path, err := syscallGetexecname()
	if err != nil {
		return path, err
	}
	if len(path) > 0 && path[0] != '/' {
		if initCwdErr != nil {
			return path, initCwdErr
		}
		if len(path) > 2 && path[0:2] == "./" {
			// skip "./"
			path = path[2:]
		}
		return initCwd + "/" + path, nil
	}
	return path, nil
}
