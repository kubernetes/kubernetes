// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !gccgo

package testtext

import "testing"

// AllocsPerRun wraps testing.AllocsPerRun.
func AllocsPerRun(runs int, f func()) (avg float64) {
	return testing.AllocsPerRun(runs, f)
}
