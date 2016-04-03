// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gensupport

import "net/url"

// URLParams is a simplified replacement for url.Values
// that safely builds up URL parameters for encoding.
type URLParams map[string][]string

// Set sets the key to value.
// It replaces any existing values.
func (u URLParams) Set(key, value string) {
	u[key] = []string{value}
}

// SetMulti sets the key to an array of values.
// It replaces any existing values.
// Note that values must not be modified after calling SetMulti
// so the caller is responsible for making a copy if necessary.
func (u URLParams) SetMulti(key string, values []string) {
	u[key] = values
}

// Encode encodes the values into ``URL encoded'' form
// ("bar=baz&foo=quux") sorted by key.
func (u URLParams) Encode() string {
	return url.Values(u).Encode()
}
