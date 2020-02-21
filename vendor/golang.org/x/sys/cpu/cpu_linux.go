// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//+build !amd64,!amd64p32,!386

package cpu

import (
	"encoding/binary"
	"io/ioutil"
	"runtime"
)

const (
	_AT_HWCAP  = 16
	_AT_HWCAP2 = 26

	procAuxv = "/proc/self/auxv"

	uintSize uint = 32 << (^uint(0) >> 63)
)

// For those platforms don't have a 'cpuid' equivalent we use HWCAP/HWCAP2
// These are initialized in cpu_$GOARCH.go
// and should not be changed after they are initialized.
var HWCap uint
var HWCap2 uint

func init() {
	buf, err := ioutil.ReadFile(procAuxv)
	if err != nil {
		panic("read proc auxv failed: " + err.Error())
	}

	pb := int(uintSize / 8)

	for i := 0; i < len(buf)-pb*2; i += pb * 2 {
		var tag, val uint
		switch uintSize {
		case 32:
			tag = uint(binary.LittleEndian.Uint32(buf[i:]))
			val = uint(binary.LittleEndian.Uint32(buf[i+pb:]))
		case 64:
			if runtime.GOARCH == "ppc64" {
				tag = uint(binary.BigEndian.Uint64(buf[i:]))
				val = uint(binary.BigEndian.Uint64(buf[i+pb:]))
			} else {
				tag = uint(binary.LittleEndian.Uint64(buf[i:]))
				val = uint(binary.LittleEndian.Uint64(buf[i+pb:]))
			}
		}
		switch tag {
		case _AT_HWCAP:
			HWCap = val
		case _AT_HWCAP2:
			HWCap2 = val
		}
	}
	doinit()
}
