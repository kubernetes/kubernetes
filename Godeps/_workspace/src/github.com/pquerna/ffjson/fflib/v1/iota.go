/**
 *  Copyright 2014 Paul Querna
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 */

/* Portions of this file are on Go stdlib's strconv/iota.go */
// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package v1

import (
	"io"
)

const (
	digits   = "0123456789abcdefghijklmnopqrstuvwxyz"
	digits01 = "0123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789"
	digits10 = "0000000000111111111122222222223333333333444444444455555555556666666666777777777788888888889999999999"
)

var shifts = [len(digits) + 1]uint{
	1 << 1: 1,
	1 << 2: 2,
	1 << 3: 3,
	1 << 4: 4,
	1 << 5: 5,
}

var smallNumbers = [][]byte{
	[]byte("0"),
	[]byte("1"),
	[]byte("2"),
	[]byte("3"),
	[]byte("4"),
	[]byte("5"),
	[]byte("6"),
	[]byte("7"),
	[]byte("8"),
	[]byte("9"),
	[]byte("10"),
}

type FormatBitsWriter interface {
	io.Writer
	io.ByteWriter
}

type FormatBitsScratch struct{}

//
// DEPRECIATED: `scratch` is no longer used, FormatBits2 is available.
//
// FormatBits computes the string representation of u in the given base.
// If neg is set, u is treated as negative int64 value. If append_ is
// set, the string is appended to dst and the resulting byte slice is
// returned as the first result value; otherwise the string is returned
// as the second result value.
//
func FormatBits(scratch *FormatBitsScratch, dst FormatBitsWriter, u uint64, base int, neg bool) {
	FormatBits2(dst, u, base, neg)
}

// FormatBits2 computes the string representation of u in the given base.
// If neg is set, u is treated as negative int64 value. If append_ is
// set, the string is appended to dst and the resulting byte slice is
// returned as the first result value; otherwise the string is returned
// as the second result value.
//
func FormatBits2(dst FormatBitsWriter, u uint64, base int, neg bool) {
	if base < 2 || base > len(digits) {
		panic("strconv: illegal AppendInt/FormatInt base")
	}
	// fast path for small common numbers
	if u <= 10 {
		if neg {
			dst.WriteByte('-')
		}
		dst.Write(smallNumbers[u])
		return
	}

	// 2 <= base && base <= len(digits)

	var a = makeSlice(65)
	//	var a [64 + 1]byte // +1 for sign of 64bit value in base 2
	i := len(a)

	if neg {
		u = -u
	}

	// convert bits
	if base == 10 {
		// common case: use constants for / and % because
		// the compiler can optimize it into a multiply+shift,
		// and unroll loop
		for u >= 100 {
			i -= 2
			q := u / 100
			j := uintptr(u - q*100)
			a[i+1] = digits01[j]
			a[i+0] = digits10[j]
			u = q
		}
		if u >= 10 {
			i--
			q := u / 10
			a[i] = digits[uintptr(u-q*10)]
			u = q
		}

	} else if s := shifts[base]; s > 0 {
		// base is power of 2: use shifts and masks instead of / and %
		b := uint64(base)
		m := uintptr(b) - 1 // == 1<<s - 1
		for u >= b {
			i--
			a[i] = digits[uintptr(u)&m]
			u >>= s
		}

	} else {
		// general case
		b := uint64(base)
		for u >= b {
			i--
			a[i] = digits[uintptr(u%b)]
			u /= b
		}
	}

	// u < base
	i--
	a[i] = digits[uintptr(u)]

	// add sign, if any
	if neg {
		i--
		a[i] = '-'
	}

	dst.Write(a[i:])

	Pool(a)

	return
}
