// Copyright 2021, The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !go1.13

// TODO(≥go1.13): For support on <go1.13, we use the xerrors package.
// Drop this file when we no longer support older Go versions.

package cmpopts

import "golang.org/x/xerrors"

func compareErrors(x, y interface{}) bool {
	xe := x.(error)
	ye := y.(error)
	return xerrors.Is(xe, ye) || xerrors.Is(ye, xe)
}
