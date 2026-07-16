// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !go1.27

package http2

import "net/http"

// Support for go.dev/issue/75500 is added in Go 1.27. In case anyone uses
// x/net with versions before Go 1.27, we return true here so that their write
// scheduler will still be the round-robin write scheduler rather than the RFC
// 9218 write scheduler. That way, older users of Go will not see a sudden
// change of behavior just from importing x/net.
//
// TODO(nsh): remove this file after x/net go.mod is at Go 1.27.
func clientPriorityDisabled(_ *http.Server) bool {
	return true
}
