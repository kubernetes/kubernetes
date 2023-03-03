// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !go1.18
// +build !go1.18

package http2

import (
	"crypto/tls"
	"net"
)

func tlsUnderlyingConn(tc *tls.Conn) net.Conn {
	return nil
}
