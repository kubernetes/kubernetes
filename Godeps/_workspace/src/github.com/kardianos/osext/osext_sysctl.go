// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin freebsd

package osext

import (
	"os"
	"path/filepath"
	"runtime"
	"syscall"
	"unsafe"
)

var initCwd, initCwdErr = os.Getwd()

func executable() (string, error) {
	var mib [4]int32
	switch runtime.GOOS {
	case "freebsd":
		mib = [4]int32{1 /* CTL_KERN */, 14 /* KERN_PROC */, 12 /* KERN_PROC_PATHNAME */, -1}
	case "darwin":
		mib = [4]int32{1 /* CTL_KERN */, 38 /* KERN_PROCARGS */, int32(os.Getpid()), -1}
	}

	n := uintptr(0)
	// Get length.
	_, _, errNum := syscall.Syscall6(syscall.SYS___SYSCTL, uintptr(unsafe.Pointer(&mib[0])), 4, 0, uintptr(unsafe.Pointer(&n)), 0, 0)
	if errNum != 0 {
		return "", errNum
	}
	if n == 0 { // This shouldn't happen.
		return "", nil
	}
	buf := make([]byte, n)
	_, _, errNum = syscall.Syscall6(syscall.SYS___SYSCTL, uintptr(unsafe.Pointer(&mib[0])), 4, uintptr(unsafe.Pointer(&buf[0])), uintptr(unsafe.Pointer(&n)), 0, 0)
	if errNum != 0 {
		return "", errNum
	}
	if n == 0 { // This shouldn't happen.
		return "", nil
	}
	for i, v := range buf {
		if v == 0 {
			buf = buf[:i]
			break
		}
	}
	var err error
	execPath := string(buf)
	// execPath will not be empty due to above checks.
	// Try to get the absolute path if the execPath is not rooted.
	if execPath[0] != '/' {
		execPath, err = getAbs(execPath)
		if err != nil {
			return execPath, err
		}
	}
	// For darwin KERN_PROCARGS may return the path to a symlink rather than the
	// actual executable.
	if runtime.GOOS == "darwin" {
		if execPath, err = filepath.EvalSymlinks(execPath); err != nil {
			return execPath, err
		}
	}
	return execPath, nil
}

func getAbs(execPath string) (string, error) {
	if initCwdErr != nil {
		return execPath, initCwdErr
	}
	// The execPath may begin with a "../" or a "./" so clean it first.
	// Join the two paths, trailing and starting slashes undetermined, so use
	// the generic Join function.
	return filepath.Join(initCwd, filepath.Clean(execPath)), nil
}
