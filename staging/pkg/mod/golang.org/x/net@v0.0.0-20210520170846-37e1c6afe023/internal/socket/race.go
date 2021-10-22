// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build race
// +build race

package socket

import (
	"runtime"
	"unsafe"
)

// This package reads and writes the Message buffers using a
// direct system call, which the race detector can't see.
// These functions tell the race detector what is going on during the syscall.

func (m *Message) raceRead() {
	for _, b := range m.Buffers {
		if len(b) > 0 {
			runtime.RaceReadRange(unsafe.Pointer(&b[0]), len(b))
		}
	}
	if b := m.OOB; len(b) > 0 {
		runtime.RaceReadRange(unsafe.Pointer(&b[0]), len(b))
	}
}
func (m *Message) raceWrite() {
	for _, b := range m.Buffers {
		if len(b) > 0 {
			runtime.RaceWriteRange(unsafe.Pointer(&b[0]), len(b))
		}
	}
	if b := m.OOB; len(b) > 0 {
		runtime.RaceWriteRange(unsafe.Pointer(&b[0]), len(b))
	}
}
