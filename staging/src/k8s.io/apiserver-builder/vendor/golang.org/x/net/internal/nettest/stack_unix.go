// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd linux netbsd openbsd solaris

package nettest

import (
	"fmt"
	"os"
	"runtime"
)

// SupportsRawIPSocket reports whether the platform supports raw IP
// sockets.
func SupportsRawIPSocket() (string, bool) {
	if os.Getuid() != 0 {
		return fmt.Sprintf("must be root on %s", runtime.GOOS), false
	}
	return "", true
}
