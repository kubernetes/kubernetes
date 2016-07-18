// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !go1.6

package http2

import "net/http"

func configureTransport(t1 *http.Transport) (*Transport, error) {
	return nil, errTransportVersion
}
