// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build aix || darwin || dragonfly || freebsd || linux || netbsd || openbsd || solaris || zos
// +build aix darwin dragonfly freebsd linux netbsd openbsd solaris zos

package ipv6_test

import (
	"fmt"
	"runtime"
)

func supportsIPv6MulticastDeliveryOnLoopback() (string, bool) {
	switch runtime.GOOS {
	case "freebsd":
		// See http://www.freebsd.org/cgi/query-pr.cgi?pr=180065.
		// Even after the fix, it looks like the latest
		// kernels don't deliver link-local scoped multicast
		// packets correctly.
		return fmt.Sprintf("not supported on %s/%s", runtime.GOOS, runtime.GOARCH), false
	default:
		return "", true
	}
}
