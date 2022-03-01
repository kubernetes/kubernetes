// Copyright 2021 Google LLC.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build go1.16

package http

import (
	"net/http"
	"time"

	"golang.org/x/net/http2"
)

// configureHTTP2 configures the ReadIdleTimeout HTTP/2 option for the
// transport. This allows broken idle connections to be pruned more quickly,
// preventing the client from attempting to re-use connections that will no
// longer work.
func configureHTTP2(trans *http.Transport) {
	http2Trans, err := http2.ConfigureTransports(trans)
	if err == nil {
		http2Trans.ReadIdleTimeout = time.Second * 15
	}
}
