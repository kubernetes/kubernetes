// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build nacl plan9

package nettest

import (
	"fmt"
	"runtime"
)

// SupportsRawIPSocket reports whether the platform supports raw IP
// sockets.
func SupportsRawIPSocket() (string, bool) {
	return fmt.Sprintf("not supported on %s", runtime.GOOS), false
}
