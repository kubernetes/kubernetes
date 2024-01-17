// Copyright 2016 The Snappy-Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !appengine
// +build gc
// +build !noasm
// +build amd64 arm64

package snappy

// emitLiteral has the same semantics as in encode_other.go.
//
//go:noescape
func emitLiteral(dst, lit []byte) int

// emitCopy has the same semantics as in encode_other.go.
//
//go:noescape
func emitCopy(dst []byte, offset, length int) int

// extendMatch has the same semantics as in encode_other.go.
//
//go:noescape
func extendMatch(src []byte, i, j int) int

// encodeBlock has the same semantics as in encode_other.go.
//
//go:noescape
func encodeBlock(dst, src []byte) (d int)
