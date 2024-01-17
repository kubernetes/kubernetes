// Copyright 2016 The Snappy-Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !appengine
// +build gc
// +build !noasm
// +build amd64 arm64

package snappy

// decode has the same semantics as in decode_other.go.
//
//go:noescape
func decode(dst, src []byte) int
