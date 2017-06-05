// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !linux

package ipv4

import "golang.org/x/net/bpf"

// SetBPF attaches a BPF program to the connection.
//
// Only supported on Linux.
func (c *dgramOpt) SetBPF(filter []bpf.RawInstruction) error {
	return errOpNoSupport
}
