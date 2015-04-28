// Copyright 2014 The lldb Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Utilities to encode/decode and collate Go predeclared scalar types (and the
// typeless nil and []byte).  The encoding format is a variation of the one
// used by the "encoding/gob" package.

package lldb

import (
	"bytes"
	"fmt"
	"math"

	"github.com/cznic/mathutil"
)

const (
	gbNull     = iota // 0x00
	gbFalse           // 0x01
	gbTrue            // 0x02
	gbFloat0          // 0x03
	gbFloat1          // 0x04
	gbFloat2          // 0x05
	gbFloat3          // 0x06
	gbFloat4          // 0x07
	gbFloat5          // 0x08
	gbFloat6          // 0x09
	gbFloat7          // 0x0a
	gbFloat8          // 0x0b
	gbComplex0        // 0x0c
	gbComplex1        // 0x0d
	gbComplex2        // 0x0e
	gbComplex3        // 0x0f
	gbComplex4        // 0x10
	gbComplex5        // 0x11
	gbComplex6        // 0x12
	gbComplex7        // 0x13
	gbComplex8        // 0x14
	gbBytes00         // 0x15
	gbBytes01         // 0x16
	gbBytes02         // 0x17
	gbBytes03         // 0x18
	gbBytes04         // 0x19
	gbBytes05         // 0x1a
	gbBytes06         // 0x1b
	gbBytes07         // 0x1c
	gbBytes08         // 0x1d
	gbBytes09         // 0x1e
	gbBytes10         // 0x1f
	gbBytes11         // 0x20
	gbBytes12         // 0x21
	gbBytes13         // 0x22
	gbBytes14         // 0x23
	gbBytes15         // 0x24
	gbBytes16         // 0x25
	gbBytes17         // Ox26
	gbBytes1          // 0x27
	gbBytes2          // 0x28: Offset by one to allow 64kB sized []byte.
	gbString00        // 0x29
	gbString01        // 0x2a
	gbString02        // 0x2b
	gbString03        // 0x2c
	gbString04        // 0x2d
	gbString05        // 0x2e
	gbString06        // 0x2f
	gbString07        // 0x30
	gbString08        // 0x31
	gbString09        // 0x32
	gbString10        // 0x33
	gbString11        // 0x34
	gbString12        // 0x35
	gbString13        // 0x36
	gbString14        // 0x37
	gbString15        // 0x38
	gbString16        // 0x39
	gbString17        // 0x3a
	gbString1         // 0x3b
	gbString2         // 0x3c
	gbUintP1          // 0x3d
	gbUintP2          // 0x3e
	gbUintP3          // 0x3f
	gbUintP4          // 0x40
	gbUintP5          // 0x41
	gbUintP6          // 0x42
	gbUintP7          // 0x43
	gbUintP8          // 0x44
	gbIntM8           // 0x45
	gbIntM7           // 0x46
	gbIntM6           // 0x47
	gbIntM5           // 0x48
	gbIntM4           // 0x49
	gbIntM3           // 0x4a
	gbIntM2           // 0x4b
	gbIntM1           // 0x4c
	gbIntP1           // 0x4d
	gbIntP2           // 0x4e
	gbIntP3           // 0x4f
	gbIntP4           // 0x50
	gbIntP5           // 0x51
	gbIntP6           // 0x52
	gbIntP7           // 0x53
	gbIntP8           // 0x54
	gbInt0            // 0x55

	gbIntMax = 255 - gbInt0 // 0xff == 170
)

// EncodeScalars encodes a vector of predeclared scalar type values to a
// []byte, making it suitable to store it as a "record" in a DB or to use it as
// a key of a BTree.
func EncodeScalars(scalars ...interface{}) (b []byte, err error) {
	for _, scalar := range scalars {
		switch x := scalar.(type) {
		default:
			return nil, &ErrINVAL{"EncodeScalars: unsupported type", fmt.Sprintf("%T in `%#v`", x, scalars)}

		case nil:
			b = append(b, gbNull)

		case bool:
			switch x {
			case false:
				b = append(b, gbFalse)
			case true:
				b = append(b, gbTrue)
			}

		case float32:
			encFloat(float64(x), &b)
		case float64:
			encFloat(x, &b)

		case complex64:
			encComplex(complex128(x), &b)
		case complex128:
			encComplex(x, &b)

		case string:
			n := len(x)
			if n <= 17 {
				b = append(b, byte(gbString00+n))
				b = append(b, []byte(x)...)
				break
			}

			if n > 65535 {
				return nil, fmt.Errorf("EncodeScalars: cannot encode string of length %d (limit 65536)", n)
			}

			pref := byte(gbString1)
			if n > 255 {
				pref++
			}
			b = append(b, pref)
			encUint0(uint64(n), &b)
			b = append(b, []byte(x)...)

		case int8:
			encInt(int64(x), &b)
		case int16:
			encInt(int64(x), &b)
		case int32:
			encInt(int64(x), &b)
		case int64:
			encInt(x, &b)
		case int:
			encInt(int64(x), &b)

		case uint8:
			encUint(uint64(x), &b)
		case uint16:
			encUint(uint64(x), &b)
		case uint32:
			encUint(uint64(x), &b)
		case uint64:
			encUint(x, &b)
		case uint:
			encUint(uint64(x), &b)
		case []byte:
			n := len(x)
			if n <= 17 {
				b = append(b, byte(gbBytes00+n))
				b = append(b, []byte(x)...)
				break
			}

			if n > 655356 {
				return nil, fmt.Errorf("EncodeScalars: cannot encode []byte of length %d (limit 65536)", n)
			}

			pref := byte(gbBytes1)
			if n > 255 {
				pref++
			}
			b = append(b, pref)
			if n <= 255 {
				b = append(b, byte(n))
			} else {
				n--
				b = append(b, byte(n>>8), byte(n))
			}
			b = append(b, x...)
		}
	}
	return
}

func encComplex(f complex128, b *[]byte) {
	encFloatPrefix(gbComplex0, real(f), b)
	encFloatPrefix(gbComplex0, imag(f), b)
}

func encFloatPrefix(prefix byte, f float64, b *[]byte) {
	u := math.Float64bits(f)
	var n uint64
	for i := 0; i < 8; i++ {
		n <<= 8
		n |= u & 0xFF
		u >>= 8
	}
	bits := mathutil.BitLenUint64(n)
	if bits == 0 {
		*b = append(*b, prefix)
		return
	}

	// 0 1 2 3 4 5 6 7 8 9
	// . 1 1 1 1 1 1 1 1 2
	encUintPrefix(prefix+1+byte((bits-1)>>3), n, b)
}

func encFloat(f float64, b *[]byte) {
	encFloatPrefix(gbFloat0, f, b)
}

func encUint0(n uint64, b *[]byte) {
	switch {
	case n <= 0xff:
		*b = append(*b, byte(n))
	case n <= 0xffff:
		*b = append(*b, byte(n>>8), byte(n))
	case n <= 0xffffff:
		*b = append(*b, byte(n>>16), byte(n>>8), byte(n))
	case n <= 0xffffffff:
		*b = append(*b, byte(n>>24), byte(n>>16), byte(n>>8), byte(n))
	case n <= 0xffffffffff:
		*b = append(*b, byte(n>>32), byte(n>>24), byte(n>>16), byte(n>>8), byte(n))
	case n <= 0xffffffffffff:
		*b = append(*b, byte(n>>40), byte(n>>32), byte(n>>24), byte(n>>16), byte(n>>8), byte(n))
	case n <= 0xffffffffffffff:
		*b = append(*b, byte(n>>48), byte(n>>40), byte(n>>32), byte(n>>24), byte(n>>16), byte(n>>8), byte(n))
	case n <= math.MaxUint64:
		*b = append(*b, byte(n>>56), byte(n>>48), byte(n>>40), byte(n>>32), byte(n>>24), byte(n>>16), byte(n>>8), byte(n))
	}
}

func encUintPrefix(prefix byte, n uint64, b *[]byte) {
	*b = append(*b, prefix)
	encUint0(n, b)
}

func encUint(n uint64, b *[]byte) {
	bits := mathutil.Max(1, mathutil.BitLenUint64(n))
	encUintPrefix(gbUintP1+byte((bits-1)>>3), n, b)
}

func encInt(n int64, b *[]byte) {
	switch {
	case n < -0x100000000000000:
		*b = append(*b, byte(gbIntM8), byte(n>>56), byte(n>>48), byte(n>>40), byte(n>>32), byte(n>>24), byte(n>>16), byte(n>>8), byte(n))
	case n < -0x1000000000000:
		*b = append(*b, byte(gbIntM7), byte(n>>48), byte(n>>40), byte(n>>32), byte(n>>24), byte(n>>16), byte(n>>8), byte(n))
	case n < -0x10000000000:
		*b = append(*b, byte(gbIntM6), byte(n>>40), byte(n>>32), byte(n>>24), byte(n>>16), byte(n>>8), byte(n))
	case n < -0x100000000:
		*b = append(*b, byte(gbIntM5), byte(n>>32), byte(n>>24), byte(n>>16), byte(n>>8), byte(n))
	case n < -0x1000000:
		*b = append(*b, byte(gbIntM4), byte(n>>24), byte(n>>16), byte(n>>8), byte(n))
	case n < -0x10000:
		*b = append(*b, byte(gbIntM3), byte(n>>16), byte(n>>8), byte(n))
	case n < -0x100:
		*b = append(*b, byte(gbIntM2), byte(n>>8), byte(n))
	case n < 0:
		*b = append(*b, byte(gbIntM1), byte(n))
	case n <= gbIntMax:
		*b = append(*b, byte(gbInt0+n))
	case n <= 0xff:
		*b = append(*b, gbIntP1, byte(n))
	case n <= 0xffff:
		*b = append(*b, gbIntP2, byte(n>>8), byte(n))
	case n <= 0xffffff:
		*b = append(*b, gbIntP3, byte(n>>16), byte(n>>8), byte(n))
	case n <= 0xffffffff:
		*b = append(*b, gbIntP4, byte(n>>24), byte(n>>16), byte(n>>8), byte(n))
	case n <= 0xffffffffff:
		*b = append(*b, gbIntP5, byte(n>>32), byte(n>>24), byte(n>>16), byte(n>>8), byte(n))
	case n <= 0xffffffffffff:
		*b = append(*b, gbIntP6, byte(n>>40), byte(n>>32), byte(n>>24), byte(n>>16), byte(n>>8), byte(n))
	case n <= 0xffffffffffffff:
		*b = append(*b, gbIntP7, byte(n>>48), byte(n>>40), byte(n>>32), byte(n>>24), byte(n>>16), byte(n>>8), byte(n))
	case n <= 0x7fffffffffffffff:
		*b = append(*b, gbIntP8, byte(n>>56), byte(n>>48), byte(n>>40), byte(n>>32), byte(n>>24), byte(n>>16), byte(n>>8), byte(n))
	}
}

func decodeFloat(b []byte) float64 {
	var u uint64
	for i, v := range b {
		u |= uint64(v) << uint((i+8-len(b))*8)
	}
	return math.Float64frombits(u)
}

// DecodeScalars decodes a []byte produced by EncodeScalars.
func DecodeScalars(b []byte) (scalars []interface{}, err error) {
	b0 := b
	for len(b) != 0 {
		switch tag := b[0]; tag {
		//default:
		//return nil, fmt.Errorf("tag %d(%#x) not supported", b[0], b[0])
		case gbNull:
			scalars = append(scalars, nil)
			b = b[1:]
		case gbFalse:
			scalars = append(scalars, false)
			b = b[1:]
		case gbTrue:
			scalars = append(scalars, true)
			b = b[1:]
		case gbFloat0:
			scalars = append(scalars, 0.0)
			b = b[1:]
		case gbFloat1, gbFloat2, gbFloat3, gbFloat4, gbFloat5, gbFloat6, gbFloat7, gbFloat8:
			n := 1 + int(tag) - gbFloat0
			if len(b) < n-1 {
				goto corrupted
			}

			scalars = append(scalars, decodeFloat(b[1:n]))
			b = b[n:]
		case gbComplex0, gbComplex1, gbComplex2, gbComplex3, gbComplex4, gbComplex5, gbComplex6, gbComplex7, gbComplex8:
			n := 1 + int(tag) - gbComplex0
			if len(b) < n-1 {
				goto corrupted
			}

			re := decodeFloat(b[1:n])
			b = b[n:]

			if len(b) == 0 {
				goto corrupted
			}

			tag = b[0]
			if tag < gbComplex0 || tag > gbComplex8 {
				goto corrupted
			}

			n = 1 + int(tag) - gbComplex0
			if len(b) < n-1 {
				goto corrupted
			}

			scalars = append(scalars, complex(re, decodeFloat(b[1:n])))
			b = b[n:]
		case gbBytes00, gbBytes01, gbBytes02, gbBytes03, gbBytes04,
			gbBytes05, gbBytes06, gbBytes07, gbBytes08, gbBytes09,
			gbBytes10, gbBytes11, gbBytes12, gbBytes13, gbBytes14,
			gbBytes15, gbBytes16, gbBytes17:
			n := int(tag - gbBytes00)
			if len(b) < n+1 {
				goto corrupted
			}

			scalars = append(scalars, append([]byte(nil), b[1:n+1]...))
			b = b[n+1:]
		case gbBytes1:
			if len(b) < 2 {
				goto corrupted
			}

			n := int(b[1])
			b = b[2:]
			if len(b) < n {
				goto corrupted
			}

			scalars = append(scalars, append([]byte(nil), b[:n]...))
			b = b[n:]
		case gbBytes2:
			if len(b) < 3 {
				goto corrupted
			}

			n := int(b[1])<<8 | int(b[2]) + 1
			b = b[3:]
			if len(b) < n {
				goto corrupted
			}

			scalars = append(scalars, append([]byte(nil), b[:n]...))
			b = b[n:]
		case gbString00, gbString01, gbString02, gbString03, gbString04,
			gbString05, gbString06, gbString07, gbString08, gbString09,
			gbString10, gbString11, gbString12, gbString13, gbString14,
			gbString15, gbString16, gbString17:
			n := int(tag - gbString00)
			if len(b) < n+1 {
				goto corrupted
			}

			scalars = append(scalars, string(b[1:n+1]))
			b = b[n+1:]
		case gbString1:
			if len(b) < 2 {
				goto corrupted
			}

			n := int(b[1])
			b = b[2:]
			if len(b) < n {
				goto corrupted
			}

			scalars = append(scalars, string(b[:n]))
			b = b[n:]
		case gbString2:
			if len(b) < 3 {
				goto corrupted
			}

			n := int(b[1])<<8 | int(b[2])
			b = b[3:]
			if len(b) < n {
				goto corrupted
			}

			scalars = append(scalars, string(b[:n]))
			b = b[n:]
		case gbUintP1, gbUintP2, gbUintP3, gbUintP4, gbUintP5, gbUintP6, gbUintP7, gbUintP8:
			b = b[1:]
			n := 1 + int(tag) - gbUintP1
			if len(b) < n {
				goto corrupted
			}

			var u uint64
			for _, v := range b[:n] {
				u = u<<8 | uint64(v)
			}
			scalars = append(scalars, u)
			b = b[n:]
		case gbIntM8, gbIntM7, gbIntM6, gbIntM5, gbIntM4, gbIntM3, gbIntM2, gbIntM1:
			b = b[1:]
			n := 8 - (int(tag) - gbIntM8)
			if len(b) < n {
				goto corrupted
			}
			u := uint64(math.MaxUint64)
			for _, v := range b[:n] {
				u = u<<8 | uint64(v)
			}
			scalars = append(scalars, int64(u))
			b = b[n:]
		case gbIntP1, gbIntP2, gbIntP3, gbIntP4, gbIntP5, gbIntP6, gbIntP7, gbIntP8:
			b = b[1:]
			n := 1 + int(tag) - gbIntP1
			if len(b) < n {
				goto corrupted
			}

			i := int64(0)
			for _, v := range b[:n] {
				i = i<<8 | int64(v)
			}
			scalars = append(scalars, i)
			b = b[n:]
		default:
			scalars = append(scalars, int64(b[0])-gbInt0)
			b = b[1:]
		}
	}
	return append([]interface{}(nil), scalars...), nil

corrupted:
	return nil, &ErrDecodeScalars{append([]byte(nil), b0...), len(b0) - len(b)}
}

func collateComplex(x, y complex128) int {
	switch rx, ry := real(x), real(y); {
	case rx < ry:
		return -1
	case rx == ry:
		switch ix, iy := imag(x), imag(y); {
		case ix < iy:
			return -1
		case ix == iy:
			return 0
		case ix > iy:
			return 1
		}
	}
	//case rx > ry:
	return 1
}

func collateFloat(x, y float64) int {
	switch {
	case x < y:
		return -1
	case x == y:
		return 0
	}
	//case x > y:
	return 1
}

func collateInt(x, y int64) int {
	switch {
	case x < y:
		return -1
	case x == y:
		return 0
	}
	//case x > y:
	return 1
}

func collateUint(x, y uint64) int {
	switch {
	case x < y:
		return -1
	case x == y:
		return 0
	}
	//case x > y:
	return 1
}

func collateIntUint(x int64, y uint64) int {
	if y > math.MaxInt64 {
		return -1
	}

	return collateInt(x, int64(y))
}

func collateUintInt(x uint64, y int64) int {
	return -collateIntUint(y, x)
}

func collateType(i interface{}) (r interface{}, err error) {
	switch x := i.(type) {
	default:
		return nil, fmt.Errorf("invalid collate type %T", x)
	case nil:
		return i, nil
	case bool:
		return i, nil
	case int8:
		return int64(x), nil
	case int16:
		return int64(x), nil
	case int32:
		return int64(x), nil
	case int64:
		return i, nil
	case int:
		return int64(x), nil
	case uint8:
		return uint64(x), nil
	case uint16:
		return uint64(x), nil
	case uint32:
		return uint64(x), nil
	case uint64:
		return i, nil
	case uint:
		return uint64(x), nil
	case float32:
		return float64(x), nil
	case float64:
		return i, nil
	case complex64:
		return complex128(x), nil
	case complex128:
		return i, nil
	case []byte:
		return i, nil
	case string:
		return i, nil
	}
}

// Collate collates two arrays of Go predeclared scalar types (and the typeless
// nil or []byte). If any other type appears in x or y, Collate will return a
// non nil error.  String items are collated using strCollate or lexically
// byte-wise (as when using Go comparison operators) when strCollate is nil.
// []byte items are collated using bytes.Compare.
//
// Collate returns:
//
// 	-1 if x <  y
// 	 0 if x == y
// 	+1 if x >  y
//
// The same value as defined above must be returned from strCollate.
//
// The "outer" ordering is: nil, bool, number, []byte, string. IOW, nil is
// "smaller" than anything else except other nil, numbers collate before
// []byte, []byte collate before strings, etc.
//
// Integers and real numbers collate as expected in math. However, complex
// numbers are not ordered in Go. Here the ordering is defined: Complex numbers
// are in comparison considered first only by their real part. Iff the result
// is equality then the imaginary part is used to determine the ordering. In
// this "second order" comparing, integers and real numbers are considered as
// complex numbers with a zero imaginary part.
func Collate(x, y []interface{}, strCollate func(string, string) int) (r int, err error) {
	nx, ny := len(x), len(y)

	switch {
	case nx == 0 && ny != 0:
		return -1, nil
	case nx == 0 && ny == 0:
		return 0, nil
	case nx != 0 && ny == 0:
		return 1, nil
	}

	r = 1
	if nx > ny {
		x, y, r = y, x, -r
	}

	var c int
	for i, xi0 := range x {
		yi0 := y[i]
		xi, err := collateType(xi0)
		if err != nil {
			return 0, err
		}

		yi, err := collateType(yi0)
		if err != nil {
			return 0, err
		}

		switch x := xi.(type) {
		default:
			panic(fmt.Errorf("internal error: %T", x))

		case nil:
			switch yi.(type) {
			case nil:
				// nop
			default:
				return -r, nil
			}

		case bool:
			switch y := yi.(type) {
			case nil:
				return r, nil
			case bool:
				switch {
				case !x && y:
					return -r, nil
				case x == y:
					// nop
				case x && !y:
					return r, nil
				}
			default:
				return -r, nil
			}

		case int64:
			switch y := yi.(type) {
			case nil, bool:
				return r, nil
			case int64:
				c = collateInt(x, y)
			case uint64:
				c = collateIntUint(x, y)
			case float64:
				c = collateFloat(float64(x), y)
			case complex128:
				c = collateComplex(complex(float64(x), 0), y)
			case []byte:
				return -r, nil
			case string:
				return -r, nil
			}

			if c != 0 {
				return c * r, nil
			}

		case uint64:
			switch y := yi.(type) {
			case nil, bool:
				return r, nil
			case int64:
				c = collateUintInt(x, y)
			case uint64:
				c = collateUint(x, y)
			case float64:
				c = collateFloat(float64(x), y)
			case complex128:
				c = collateComplex(complex(float64(x), 0), y)
			case []byte:
				return -r, nil
			case string:
				return -r, nil
			}

			if c != 0 {
				return c * r, nil
			}

		case float64:
			switch y := yi.(type) {
			case nil, bool:
				return r, nil
			case int64:
				c = collateFloat(x, float64(y))
			case uint64:
				c = collateFloat(x, float64(y))
			case float64:
				c = collateFloat(x, y)
			case complex128:
				c = collateComplex(complex(x, 0), y)
			case []byte:
				return -r, nil
			case string:
				return -r, nil
			}

			if c != 0 {
				return c * r, nil
			}

		case complex128:
			switch y := yi.(type) {
			case nil, bool:
				return r, nil
			case int64:
				c = collateComplex(x, complex(float64(y), 0))
			case uint64:
				c = collateComplex(x, complex(float64(y), 0))
			case float64:
				c = collateComplex(x, complex(y, 0))
			case complex128:
				c = collateComplex(x, y)
			case []byte:
				return -r, nil
			case string:
				return -r, nil
			}

			if c != 0 {
				return c * r, nil
			}

		case []byte:
			switch y := yi.(type) {
			case nil, bool, int64, uint64, float64, complex128:
				return r, nil
			case []byte:
				c = bytes.Compare(x, y)
			case string:
				return -r, nil
			}

			if c != 0 {
				return c * r, nil
			}

		case string:
			switch y := yi.(type) {
			case nil, bool, int64, uint64, float64, complex128:
				return r, nil
			case []byte:
				return r, nil
			case string:
				switch {
				case strCollate != nil:
					c = strCollate(x, y)
				case x < y:
					return -r, nil
				case x == y:
					c = 0
				case x > y:
					return r, nil
				}
			}

			if c != 0 {
				return c * r, nil
			}
		}
	}

	if nx == ny {
		return 0, nil
	}

	return -r, nil
}
