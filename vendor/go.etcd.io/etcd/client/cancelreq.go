// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// borrowed from golang/net/context/ctxhttp/cancelreq.go

package client

import "net/http"

func requestCanceler(tr CancelableTransport, req *http.Request) func() {
	ch := make(chan struct{})
	req.Cancel = ch

	return func() {
		close(ch)
	}
}
