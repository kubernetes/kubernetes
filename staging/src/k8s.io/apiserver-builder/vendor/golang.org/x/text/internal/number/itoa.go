// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// TODO: use build tags once a low-level public API has been established in
// package strconv.

package number

const (
	digits = "0123456789abcdefghijklmnopqrstuvwxyz"
)

var shifts = [len(digits) + 1]uint{
	1 << 1: 1,
	1 << 2: 2,
	1 << 3: 3,
	1 << 4: 4,
	1 << 5: 5,
}

// formatBits computes the string representation of u in the given base.
// If neg is set, u is treated as negative int64 value. If append_ is
// set, the string is appended to dst and the resulting byte slice is
// returned as the first result value; otherwise the string is returned
// as the second result value.
//
func formatBits(dst []byte, u uint64, base int, neg, append_ bool) (d []byte, s string) {
	if base < 2 || base > len(digits) {
		panic("strconv: illegal AppendInt/FormatInt base")
	}
	// 2 <= base && base <= len(digits)

	var a [64 + 1]byte // +1 for sign of 64bit value in base 2
	i := len(a)

	if neg {
		u = -u
	}

	// convert bits
	if base == 10 {
		// common case: use constants for / because
		// the compiler can optimize it into a multiply+shift

		if ^uintptr(0)>>32 == 0 {
			for u > uint64(^uintptr(0)) {
				q := u / 1e9
				us := uintptr(u - q*1e9) // us % 1e9 fits into a uintptr
				for j := 9; j > 0; j-- {
					i--
					qs := us / 10
					a[i] = byte(us - qs*10 + '0')
					us = qs
				}
				u = q
			}
		}

		// u guaranteed to fit into a uintptr
		us := uintptr(u)
		for us >= 10 {
			i--
			q := us / 10
			a[i] = byte(us - q*10 + '0')
			us = q
		}
		// u < 10
		i--
		a[i] = byte(us + '0')

	} else if s := shifts[base]; s > 0 {
		// base is power of 2: use shifts and masks instead of / and %
		b := uint64(base)
		m := uintptr(b) - 1 // == 1<<s - 1
		for u >= b {
			i--
			a[i] = digits[uintptr(u)&m]
			u >>= s
		}
		// u < base
		i--
		a[i] = digits[uintptr(u)]

	} else {
		// general case
		b := uint64(base)
		for u >= b {
			i--
			q := u / b
			a[i] = digits[uintptr(u-q*b)]
			u = q
		}
		// u < base
		i--
		a[i] = digits[uintptr(u)]
	}

	// add sign, if any
	if neg {
		i--
		a[i] = '-'
	}

	if append_ {
		d = append(dst, a[i:]...)
		return
	}
	s = string(a[i:])
	return
}
