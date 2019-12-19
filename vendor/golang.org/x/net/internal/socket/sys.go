// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package socket

import (
	"encoding/binary"
	"unsafe"
)

var (
	// NativeEndian is the machine native endian implementation of
	// ByteOrder.
	NativeEndian binary.ByteOrder

	kernelAlign int
)

func init() {
	i := uint32(1)
	b := (*[4]byte)(unsafe.Pointer(&i))
	if b[0] == 1 {
		NativeEndian = binary.LittleEndian
	} else {
		NativeEndian = binary.BigEndian
	}
	kernelAlign = probeProtocolStack()
}

func roundup(l int) int {
	return (l + kernelAlign - 1) &^ (kernelAlign - 1)
}
