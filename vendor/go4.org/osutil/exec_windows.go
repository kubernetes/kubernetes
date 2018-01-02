/*
Copyright 2015 The go4 Authors

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

package osutil

import (
	"path/filepath"
	"syscall"
	"unsafe"
)

var (
	modkernel32            = syscall.MustLoadDLL("kernel32.dll")
	procGetModuleFileNameW = modkernel32.MustFindProc("GetModuleFileNameW")
)

func getModuleFileName(handle syscall.Handle) (string, error) {
	n := uint32(1024)
	var buf []uint16
	for {
		buf = make([]uint16, n)
		r, err := syscallGetModuleFileName(handle, &buf[0], n)
		if err != nil {
			return "", err
		}
		if r < n {
			break
		}
		// r == n means n not big enough
		n += 1024
	}
	return syscall.UTF16ToString(buf), nil
}

func executable() (string, error) {
	p, err := getModuleFileName(0)
	return filepath.Clean(p), err
}

func syscallGetModuleFileName(module syscall.Handle, fn *uint16, len uint32) (n uint32, err error) {
	r0, _, e1 := syscall.Syscall(procGetModuleFileNameW.Addr(), 3, uintptr(module), uintptr(unsafe.Pointer(fn)), uintptr(len))
	n = uint32(r0)
	if n == 0 {
		if e1 != 0 {
			err = error(e1)
		} else {
			err = syscall.EINVAL
		}
	}
	return
}
