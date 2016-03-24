// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// borrowed from golang/net/context/ctxhttp/cancelreq.go

// +build go1.5

// Package httputil provides HTTP utility functions.
package httputil

import "net/http"

func RequestCanceler(rt http.RoundTripper, req *http.Request) func() {
	ch := make(chan struct{})
	req.Cancel = ch

	return func() {
		close(ch)
	}
}
