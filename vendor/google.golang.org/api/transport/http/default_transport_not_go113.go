// Copyright 2020 Google LLC.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !go1.13
// +build !go1.13

package http

import "net/http"

// clonedTransport returns the given RoundTripper as a cloned *http.Transport.
// For versions of Go <1.13, this is not supported, so return nil.
func clonedTransport(rt http.RoundTripper) *http.Transport {
	return nil
}
