// Copyright 2012 The win Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build windows

package win

type VAR_BSTR struct {
	vt        VARTYPE
	reserved1 [6]byte
	bstrVal   *uint16 /*BSTR*/
}
