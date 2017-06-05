// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package icmp

import (
	"encoding/binary"
	"unsafe"
)

var (
	// See http://www.freebsd.org/doc/en/books/porters-handbook/freebsd-versions.html.
	freebsdVersion uint32

	nativeEndian binary.ByteOrder
)

func init() {
	i := uint32(1)
	b := (*[4]byte)(unsafe.Pointer(&i))
	if b[0] == 1 {
		nativeEndian = binary.LittleEndian
	} else {
		nativeEndian = binary.BigEndian
	}
}
