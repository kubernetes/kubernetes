// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http2

import (
	"context"
	"net/http"
)

func (cc *ClientConn) RoundTrip(req *http.Request) (*http.Response, error) {
	return cc.roundTrip(req)
}

// SetDoNotReuse marks cc as not reusable for future HTTP requests.
func (cc *ClientConn) SetDoNotReuse() {
	cc.setDoNotReuse()
}

// CanTakeNewRequest reports whether the connection can take a new request,
// meaning it has not been closed or received or sent a GOAWAY.
//
// If the caller is going to immediately make a new request on this
// connection, use ReserveNewRequest instead.
func (cc *ClientConn) CanTakeNewRequest() bool {
	return cc.canTakeNewRequest()
}

// ReserveNewRequest is like CanTakeNewRequest but also reserves a
// concurrent stream in cc. The reservation is decremented on the
// next call to RoundTrip.
func (cc *ClientConn) ReserveNewRequest() bool {
	return cc.reserveNewRequest()
}

// State returns a snapshot of cc's state.
func (cc *ClientConn) State() ClientConnState {
	return cc.state()
}

// Shutdown gracefully closes the client connection, waiting for running streams to complete.
func (cc *ClientConn) Shutdown(ctx context.Context) error {
	return cc.shutdown(ctx)
}

// Close closes the client connection immediately.
//
// In-flight requests are interrupted. For a graceful shutdown, use Shutdown instead.
func (cc *ClientConn) Close() error {
	return cc.close()
}

// Ping sends a PING frame to the server and waits for the ack.
func (cc *ClientConn) Ping(ctx context.Context) error {
	return cc.ping(ctx)
}
