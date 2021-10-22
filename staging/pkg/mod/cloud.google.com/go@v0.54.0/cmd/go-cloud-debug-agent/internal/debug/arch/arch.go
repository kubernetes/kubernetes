// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package arch contains architecture-specific definitions.
package arch

import (
	"encoding/binary"
	"math"
)

const MaxBreakpointSize = 4 // TODO

// Architecture defines the architecture-specific details for a given machine.
type Architecture struct {
	// BreakpointSize is the size of a breakpoint instruction, in bytes.
	BreakpointSize int
	// IntSize is the size of the int type, in bytes.
	IntSize int
	// PointerSize is the size of a pointer, in bytes.
	PointerSize int
	// ByteOrder is the byte order for ints and pointers.
	ByteOrder binary.ByteOrder
	// FloatByteOrder is the byte order for floats.
	FloatByteOrder  binary.ByteOrder
	BreakpointInstr [MaxBreakpointSize]byte
}

func (a *Architecture) Int(buf []byte) int64 {
	return int64(a.Uint(buf))
}

func (a *Architecture) Uint(buf []byte) uint64 {
	if len(buf) != a.IntSize {
		panic("bad IntSize")
	}
	switch a.IntSize {
	case 4:
		return uint64(a.ByteOrder.Uint32(buf[:4]))
	case 8:
		return a.ByteOrder.Uint64(buf[:8])
	}
	panic("no IntSize")
}

func (a *Architecture) Int16(buf []byte) int16 {
	return int16(a.Uint16(buf))
}

func (a *Architecture) Int32(buf []byte) int32 {
	return int32(a.Uint32(buf))
}

func (a *Architecture) Int64(buf []byte) int64 {
	return int64(a.Uint64(buf))
}

func (a *Architecture) Uint16(buf []byte) uint16 {
	return a.ByteOrder.Uint16(buf)
}

func (a *Architecture) Uint32(buf []byte) uint32 {
	return a.ByteOrder.Uint32(buf)
}

func (a *Architecture) Uint64(buf []byte) uint64 {
	return a.ByteOrder.Uint64(buf)
}

func (a *Architecture) IntN(buf []byte) int64 {
	if len(buf) == 0 {
		return 0
	}
	x := int64(0)
	if a.ByteOrder == binary.LittleEndian {
		i := len(buf) - 1
		x = int64(int8(buf[i])) // sign-extended
		for i--; i >= 0; i-- {
			x <<= 8
			x |= int64(buf[i]) // not sign-extended
		}
	} else {
		x = int64(int8(buf[0])) // sign-extended
		for i := 1; i < len(buf); i++ {
			x <<= 8
			x |= int64(buf[i]) // not sign-extended
		}
	}
	return x
}

func (a *Architecture) UintN(buf []byte) uint64 {
	u := uint64(0)
	if a.ByteOrder == binary.LittleEndian {
		shift := uint(0)
		for _, c := range buf {
			u |= uint64(c) << shift
			shift += 8
		}
	} else {
		for _, c := range buf {
			u <<= 8
			u |= uint64(c)
		}
	}
	return u
}

func (a *Architecture) Uintptr(buf []byte) uint64 {
	if len(buf) != a.PointerSize {
		panic("bad PointerSize")
	}
	switch a.PointerSize {
	case 4:
		return uint64(a.ByteOrder.Uint32(buf[:4]))
	case 8:
		return a.ByteOrder.Uint64(buf[:8])
	}
	panic("no PointerSize")
}

func (a *Architecture) Float32(buf []byte) float32 {
	if len(buf) != 4 {
		panic("bad float32 size")
	}
	return math.Float32frombits(a.FloatByteOrder.Uint32(buf))
}

func (a *Architecture) Float64(buf []byte) float64 {
	if len(buf) != 8 {
		panic("bad float64 size")
	}
	return math.Float64frombits(a.FloatByteOrder.Uint64(buf))
}

func (a *Architecture) Complex64(buf []byte) complex64 {
	if len(buf) != 8 {
		panic("bad complex64 size")
	}
	return complex(a.Float32(buf[0:4]), a.Float32(buf[4:8]))
}

func (a *Architecture) Complex128(buf []byte) complex128 {
	if len(buf) != 16 {
		panic("bad complex128 size")
	}
	return complex(a.Float64(buf[0:8]), a.Float64(buf[8:16]))
}

var AMD64 = Architecture{
	BreakpointSize:  1,
	IntSize:         8,
	PointerSize:     8,
	ByteOrder:       binary.LittleEndian,
	FloatByteOrder:  binary.LittleEndian,
	BreakpointInstr: [MaxBreakpointSize]byte{0xCC}, // INT 3
}

var X86 = Architecture{
	BreakpointSize:  1,
	IntSize:         4,
	PointerSize:     4,
	ByteOrder:       binary.LittleEndian,
	FloatByteOrder:  binary.LittleEndian,
	BreakpointInstr: [MaxBreakpointSize]byte{0xCC}, // INT 3
}

var ARM = Architecture{
	BreakpointSize:  4, // TODO
	IntSize:         4,
	PointerSize:     4,
	ByteOrder:       binary.LittleEndian,
	FloatByteOrder:  binary.LittleEndian,                             // TODO
	BreakpointInstr: [MaxBreakpointSize]byte{0x00, 0x00, 0x00, 0x00}, // TODO
}
