// Copyright 2021 Google LLC.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !go1.16
// +build !go1.16

package http

import (
	"net/http"
)

// configureHTTP2 configures the ReadIdleTimeout HTTP/2 option for the
// transport. The interface to do this is only available in Go 1.16 and up, so
// this performs a no-op.
func configureHTTP2(trans *http.Transport) {}
