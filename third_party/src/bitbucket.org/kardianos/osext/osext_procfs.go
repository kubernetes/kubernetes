// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build linux netbsd openbsd

package osext

import (
	"errors"
	"os"
	"runtime"
)

func executable() (string, error) {
	switch runtime.GOOS {
	case "linux":
		return os.Readlink("/proc/self/exe")
	case "netbsd":
		return os.Readlink("/proc/curproc/exe")
	case "openbsd":
		return os.Readlink("/proc/curproc/file")
	}
	return "", errors.New("ExecPath not implemented for " + runtime.GOOS)
}
