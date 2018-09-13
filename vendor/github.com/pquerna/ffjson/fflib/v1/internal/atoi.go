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

/* Portions of this file are on Go stdlib's strconv/atoi.go */
// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package internal

import (
	"errors"
	"strconv"
)

// ErrRange indicates that a value is out of range for the target type.
var ErrRange = errors.New("value out of range")

// ErrSyntax indicates that a value does not have the right syntax for the target type.
var ErrSyntax = errors.New("invalid syntax")

// A NumError records a failed conversion.
type NumError struct {
	Func string // the failing function (ParseBool, ParseInt, ParseUint, ParseFloat)
	Num  string // the input
	Err  error  // the reason the conversion failed (ErrRange, ErrSyntax)
}

func (e *NumError) Error() string {
	return "strconv." + e.Func + ": " + "parsing " + strconv.Quote(e.Num) + ": " + e.Err.Error()
}

func syntaxError(fn, str string) *NumError {
	return &NumError{fn, str, ErrSyntax}
}

func rangeError(fn, str string) *NumError {
	return &NumError{fn, str, ErrRange}
}

const intSize = 32 << uint(^uint(0)>>63)

// IntSize is the size in bits of an int or uint value.
const IntSize = intSize

// Return the first number n such that n*base >= 1<<64.
func cutoff64(base int) uint64 {
	if base < 2 {
		return 0
	}
	return (1<<64-1)/uint64(base) + 1
}

// ParseUint is like ParseInt but for unsigned numbers, and oeprating on []byte
func ParseUint(s []byte, base int, bitSize int) (n uint64, err error) {
	var cutoff, maxVal uint64

	if bitSize == 0 {
		bitSize = int(IntSize)
	}

	s0 := s
	switch {
	case len(s) < 1:
		err = ErrSyntax
		goto Error

	case 2 <= base && base <= 36:
		// valid base; nothing to do

	case base == 0:
		// Look for octal, hex prefix.
		switch {
		case s[0] == '0' && len(s) > 1 && (s[1] == 'x' || s[1] == 'X'):
			base = 16
			s = s[2:]
			if len(s) < 1 {
				err = ErrSyntax
				goto Error
			}
		case s[0] == '0':
			base = 8
		default:
			base = 10
		}

	default:
		err = errors.New("invalid base " + strconv.Itoa(base))
		goto Error
	}

	n = 0
	cutoff = cutoff64(base)
	maxVal = 1<<uint(bitSize) - 1

	for i := 0; i < len(s); i++ {
		var v byte
		d := s[i]
		switch {
		case '0' <= d && d <= '9':
			v = d - '0'
		case 'a' <= d && d <= 'z':
			v = d - 'a' + 10
		case 'A' <= d && d <= 'Z':
			v = d - 'A' + 10
		default:
			n = 0
			err = ErrSyntax
			goto Error
		}
		if int(v) >= base {
			n = 0
			err = ErrSyntax
			goto Error
		}

		if n >= cutoff {
			// n*base overflows
			n = 1<<64 - 1
			err = ErrRange
			goto Error
		}
		n *= uint64(base)

		n1 := n + uint64(v)
		if n1 < n || n1 > maxVal {
			// n+v overflows
			n = 1<<64 - 1
			err = ErrRange
			goto Error
		}
		n = n1
	}

	return n, nil

Error:
	return n, &NumError{"ParseUint", string(s0), err}
}

// ParseInt interprets a string s in the given base (2 to 36) and
// returns the corresponding value i.  If base == 0, the base is
// implied by the string's prefix: base 16 for "0x", base 8 for
// "0", and base 10 otherwise.
//
// The bitSize argument specifies the integer type
// that the result must fit into.  Bit sizes 0, 8, 16, 32, and 64
// correspond to int, int8, int16, int32, and int64.
//
// The errors that ParseInt returns have concrete type *NumError
// and include err.Num = s.  If s is empty or contains invalid
// digits, err.Err = ErrSyntax and the returned value is 0;
// if the value corresponding to s cannot be represented by a
// signed integer of the given size, err.Err = ErrRange and the
// returned value is the maximum magnitude integer of the
// appropriate bitSize and sign.
func ParseInt(s []byte, base int, bitSize int) (i int64, err error) {
	const fnParseInt = "ParseInt"

	if bitSize == 0 {
		bitSize = int(IntSize)
	}

	// Empty string bad.
	if len(s) == 0 {
		return 0, syntaxError(fnParseInt, string(s))
	}

	// Pick off leading sign.
	s0 := s
	neg := false
	if s[0] == '+' {
		s = s[1:]
	} else if s[0] == '-' {
		neg = true
		s = s[1:]
	}

	// Convert unsigned and check range.
	var un uint64
	un, err = ParseUint(s, base, bitSize)
	if err != nil && err.(*NumError).Err != ErrRange {
		err.(*NumError).Func = fnParseInt
		err.(*NumError).Num = string(s0)
		return 0, err
	}
	cutoff := uint64(1 << uint(bitSize-1))
	if !neg && un >= cutoff {
		return int64(cutoff - 1), rangeError(fnParseInt, string(s0))
	}
	if neg && un > cutoff {
		return -int64(cutoff), rangeError(fnParseInt, string(s0))
	}
	n := int64(un)
	if neg {
		n = -n
	}
	return n, nil
}
