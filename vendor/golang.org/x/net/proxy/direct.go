// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package proxy

import (
	"context"
	"net"
)

type direct struct{}

// Direct implements Dialer by making network connections directly using net.Dial or net.DialContext.
var Direct = direct{}

var (
	_ Dialer        = Direct
	_ ContextDialer = Direct
)

// Dial directly invokes net.Dial with the supplied parameters.
func (direct) Dial(network, addr string) (net.Conn, error) {
	return net.Dial(network, addr)
}

// DialContext instantiates a net.Dialer and invokes its DialContext receiver with the supplied parameters.
func (direct) DialContext(ctx context.Context, network, addr string) (net.Conn, error) {
	var d net.Dialer
	return d.DialContext(ctx, network, addr)
}
