// Copyright 2020 Google LLC.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.13
// +build go1.13

package http

import "net/http"

// clonedTransport returns the given RoundTripper as a cloned *http.Transport.
// It returns nil if the RoundTripper can't be cloned or coerced to
// *http.Transport.
func clonedTransport(rt http.RoundTripper) *http.Transport {
	t, ok := rt.(*http.Transport)
	if !ok {
		return nil
	}
	return t.Clone()
}
