// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !go1.8

package http2

import (
	"io"
	"net/http"
)

func configureServer18(h1 *http.Server, h2 *Server) error {
	// No IdleTimeout to sync prior to Go 1.8.
	return nil
}

func shouldLogPanic(panicValue interface{}) bool {
	return panicValue != nil
}

func reqGetBody(req *http.Request) func() (io.ReadCloser, error) {
	return nil
}

func reqBodyIsNoBody(io.ReadCloser) bool { return false }

func go18httpNoBody() io.ReadCloser { return nil } // for tests only
