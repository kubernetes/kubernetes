// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd netbsd openbsd

package nettest

import (
	"runtime"
)

func supportsIPv6MulticastDeliveryOnLoopback() bool {
	switch runtime.GOOS {
	case "freebsd":
		// See http://www.freebsd.org/cgi/query-pr.cgi?pr=180065.
		// Even after the fix, it looks like the latest
		// kernels don't deliver link-local scoped multicast
		// packets correctly.
		return false
	default:
		return true
	}
}
