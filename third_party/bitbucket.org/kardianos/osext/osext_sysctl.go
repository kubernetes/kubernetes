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

var startUpcwd, getwdError = os.Getwd()

func executable() (string, error) {
	var mib [4]int32
	switch runtime.GOOS {
	case "freebsd":
		mib = [4]int32{1 /* CTL_KERN */, 14 /* KERN_PROC */, 12 /* KERN_PROC_PATHNAME */, -1}
	case "darwin":
		mib = [4]int32{1 /* CTL_KERN */, 38 /* KERN_PROCARGS */, int32(os.Getpid()), -1}
	}

	n := uintptr(0)
	// get length
	_, _, err := syscall.Syscall6(syscall.SYS___SYSCTL, uintptr(unsafe.Pointer(&mib[0])), 4, 0, uintptr(unsafe.Pointer(&n)), 0, 0)
	if err != 0 {
		return "", err
	}
	if n == 0 { // shouldn't happen
		return "", nil
	}
	buf := make([]byte, n)
	_, _, err = syscall.Syscall6(syscall.SYS___SYSCTL, uintptr(unsafe.Pointer(&mib[0])), 4, uintptr(unsafe.Pointer(&buf[0])), uintptr(unsafe.Pointer(&n)), 0, 0)
	if err != 0 {
		return "", err
	}
	if n == 0 { // shouldn't happen
		return "", nil
	}
	for i, v := range buf {
		if v == 0 {
			buf = buf[:i]
			break
		}
	}
	var strpath string
	if buf[0] != '/' {
		var e error
		if strpath, e = getAbs(buf); e != nil {
			return strpath, e
		}
	} else {
		strpath = string(buf)
	}
	// darwin KERN_PROCARGS may return the path to a symlink rather than the
	// actual executable
	if runtime.GOOS == "darwin" {
		if strpath, err := filepath.EvalSymlinks(strpath); err != nil {
			return strpath, err
		}
	}
	return strpath, nil
}

func getAbs(buf []byte) (string, error) {
	if getwdError != nil {
		return string(buf), getwdError
	} else {
		if buf[0] == '.' {
			buf = buf[1:]
		}
		if startUpcwd[len(startUpcwd)-1] != '/' && buf[0] != '/' {
			return startUpcwd + "/" + string(buf), nil
		}
		return startUpcwd + string(buf), nil
	}
}
