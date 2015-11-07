// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// borrowed from golang/net/context/ctxhttp/cancelreq_go14.go

// +build !go1.5

package client

import "net/http"

func requestCanceler(tr CancelableTransport, req *http.Request) func() {
	return func() {
		tr.CancelRequest(req)
	}
}
