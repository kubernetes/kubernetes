// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !go1.5

package ctxhttp

import "net/http"

type requestCanceler interface {
	CancelRequest(*http.Request)
}

func canceler(client *http.Client, req *http.Request) func() {
	rc, ok := client.Transport.(requestCanceler)
	if !ok {
		return func() {}
	}
	return func() {
		rc.CancelRequest(req)
	}
}
