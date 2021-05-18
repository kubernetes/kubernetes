// Copyright 2017, The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE.md file.

// +build !cmp_debug

package diff

var debug debugger

type debugger struct{}

func (debugger) Begin(_, _ int, f EqualFunc, _, _ *EditScript) EqualFunc {
	return f
}
func (debugger) Update() {}
func (debugger) Finish() {}
