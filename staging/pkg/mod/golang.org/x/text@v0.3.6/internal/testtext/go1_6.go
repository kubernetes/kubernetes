// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !go1.7
// +build !go1.7

package testtext

import "testing"

func Run(t *testing.T, name string, fn func(t *testing.T)) bool {
	t.Logf("Running %s...", name)
	fn(t)
	return t.Failed()
}

// Bench runs the given benchmark function. This pre-1.7 implementation renders
// the measurement useless, but allows the code to be compiled at least.
func Bench(b *testing.B, name string, fn func(b *testing.B)) bool {
	b.Logf("Running %s...", name)
	fn(b)
	return b.Failed()
}
