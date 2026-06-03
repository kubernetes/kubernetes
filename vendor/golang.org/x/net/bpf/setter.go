// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bpf

// A Setter is a type which can attach a compiled BPF filter to itself.
type Setter interface {
	SetBPF(filter []RawInstruction) error
}
