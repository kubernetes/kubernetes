// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !go1.24

package http2

import "net/http"

// Pre-Go 1.24 fallback.
// The Server.HTTP2 and Transport.HTTP2 config fields were added in Go 1.24.

func fillNetHTTPServerConfig(conf *http2Config, srv *http.Server) {}

func fillNetHTTPTransportConfig(conf *http2Config, tr *http.Transport) {}
