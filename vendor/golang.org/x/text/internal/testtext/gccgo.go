// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build gccgo

package testtext

// AllocsPerRun always returns 0 for gccgo until gccgo implements escape
// analysis equal or better to that of gc.
func AllocsPerRun(runs int, f func()) (avg float64) { return 0 }
