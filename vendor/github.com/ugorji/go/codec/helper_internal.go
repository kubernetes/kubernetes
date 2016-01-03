// Copyright (c) 2012-2015 Ugorji Nwoke. All rights reserved.
// Use of this source code is governed by a MIT license found in the LICENSE file.

package codec

// All non-std package dependencies live in this file,
// so porting to different environment is easy (just update functions).

import (
	"errors"
	"fmt"
	"math"
	"reflect"
)

func panicValToErr(panicVal interface{}, err *error) {
	if panicVal == nil {
		return
	}
	// case nil
	switch xerr := panicVal.(type) {
	case error:
		*err = xerr
	case string:
		*err = errors.New(xerr)
	default:
		*err = fmt.Errorf("%v", panicVal)
	}
	return
}

func hIsEmptyValue(v reflect.Value, deref, checkStruct bool) bool {
	switch v.Kind() {
	case reflect.Invalid:
		return true
	case reflect.Array, reflect.Map, reflect.Slice, reflect.String:
		return v.Len() == 0
	case reflect.Bool:
		return !v.Bool()
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return v.Int() == 0
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		return v.Uint() == 0
	case reflect.Float32, reflect.Float64:
		return v.Float() == 0
	case reflect.Interface, reflect.Ptr:
		if deref {
			if v.IsNil() {
				return true
			}
			return hIsEmptyValue(v.Elem(), deref, checkStruct)
		} else {
			return v.IsNil()
		}
	case reflect.Struct:
		if !checkStruct {
			return false
		}
		// return true if all fields are empty. else return false.
		// we cannot use equality check, because some fields may be maps/slices/etc
		// and consequently the structs are not comparable.
		// return v.Interface() == reflect.Zero(v.Type()).Interface()
		for i, n := 0, v.NumField(); i < n; i++ {
			if !hIsEmptyValue(v.Field(i), deref, checkStruct) {
				return false
			}
		}
		return true
	}
	return false
}

func isEmptyValue(v reflect.Value) bool {
	return hIsEmptyValue(v, derefForIsEmptyValue, checkStructForEmptyValue)
}

func pruneSignExt(v []byte, pos bool) (n int) {
	if len(v) < 2 {
	} else if pos && v[0] == 0 {
		for ; v[n] == 0 && n+1 < len(v) && (v[n+1]&(1<<7) == 0); n++ {
		}
	} else if !pos && v[0] == 0xff {
		for ; v[n] == 0xff && n+1 < len(v) && (v[n+1]&(1<<7) != 0); n++ {
		}
	}
	return
}

func implementsIntf(typ, iTyp reflect.Type) (success bool, indir int8) {
	if typ == nil {
		return
	}
	rt := typ
	// The type might be a pointer and we need to keep
	// dereferencing to the base type until we find an implementation.
	for {
		if rt.Implements(iTyp) {
			return true, indir
		}
		if p := rt; p.Kind() == reflect.Ptr {
			indir++
			if indir >= math.MaxInt8 { // insane number of indirections
				return false, 0
			}
			rt = p.Elem()
			continue
		}
		break
	}
	// No luck yet, but if this is a base type (non-pointer), the pointer might satisfy.
	if typ.Kind() != reflect.Ptr {
		// Not a pointer, but does the pointer work?
		if reflect.PtrTo(typ).Implements(iTyp) {
			return true, -1
		}
	}
	return false, 0
}

// validate that this function is correct ...
// culled from OGRE (Object-Oriented Graphics Rendering Engine)
// function: halfToFloatI (http://stderr.org/doc/ogre-doc/api/OgreBitwise_8h-source.html)
func halfFloatToFloatBits(yy uint16) (d uint32) {
	y := uint32(yy)
	s := (y >> 15) & 0x01
	e := (y >> 10) & 0x1f
	m := y & 0x03ff

	if e == 0 {
		if m == 0 { // plu or minus 0
			return s << 31
		} else { // Denormalized number -- renormalize it
			for (m & 0x00000400) == 0 {
				m <<= 1
				e -= 1
			}
			e += 1
			const zz uint32 = 0x0400
			m &= ^zz
		}
	} else if e == 31 {
		if m == 0 { // Inf
			return (s << 31) | 0x7f800000
		} else { // NaN
			return (s << 31) | 0x7f800000 | (m << 13)
		}
	}
	e = e + (127 - 15)
	m = m << 13
	return (s << 31) | (e << 23) | m
}

// GrowCap will return a new capacity for a slice, given the following:
//   - oldCap: current capacity
//   - unit: in-memory size of an element
//   - num: number of elements to add
func growCap(oldCap, unit, num int) (newCap int) {
	// appendslice logic (if cap < 1024, *2, else *1.25):
	//   leads to many copy calls, especially when copying bytes.
	//   bytes.Buffer model (2*cap + n): much better for bytes.
	// smarter way is to take the byte-size of the appended element(type) into account

	// maintain 3 thresholds:
	// t1: if cap <= t1, newcap = 2x
	// t2: if cap <= t2, newcap = 1.75x
	// t3: if cap <= t3, newcap = 1.5x
	//     else          newcap = 1.25x
	//
	// t1, t2, t3 >= 1024 always.
	// i.e. if unit size >= 16, then always do 2x or 1.25x (ie t1, t2, t3 are all same)
	//
	// With this, appending for bytes increase by:
	//    100% up to 4K
	//     75% up to 8K
	//     50% up to 16K
	//     25% beyond that

	// unit can be 0 e.g. for struct{}{}; handle that appropriately
	var t1, t2, t3 int // thresholds
	if unit <= 1 {
		t1, t2, t3 = 4*1024, 8*1024, 16*1024
	} else if unit < 16 {
		t3 = 16 / unit * 1024
		t1 = t3 * 1 / 4
		t2 = t3 * 2 / 4
	} else {
		t1, t2, t3 = 1024, 1024, 1024
	}

	var x int // temporary variable

	// x is multiplier here: one of 5, 6, 7 or 8; incr of 25%, 50%, 75% or 100% respectively
	if oldCap <= t1 { // [0,t1]
		x = 8
	} else if oldCap > t3 { // (t3,infinity]
		x = 5
	} else if oldCap <= t2 { // (t1,t2]
		x = 7
	} else { // (t2,t3]
		x = 6
	}
	newCap = x * oldCap / 4

	if num > 0 {
		newCap += num
	}

	// ensure newCap is a multiple of 64 (if it is > 64) or 16.
	if newCap > 64 {
		if x = newCap % 64; x != 0 {
			x = newCap / 64
			newCap = 64 * (x + 1)
		}
	} else {
		if x = newCap % 16; x != 0 {
			x = newCap / 16
			newCap = 16 * (x + 1)
		}
	}
	return
}

func expandSliceValue(s reflect.Value, num int) reflect.Value {
	if num <= 0 {
		return s
	}
	l0 := s.Len()
	l1 := l0 + num // new slice length
	if l1 < l0 {
		panic("ExpandSlice: slice overflow")
	}
	c0 := s.Cap()
	if l1 <= c0 {
		return s.Slice(0, l1)
	}
	st := s.Type()
	c1 := growCap(c0, int(st.Elem().Size()), num)
	s2 := reflect.MakeSlice(st, l1, c1)
	// println("expandslicevalue: cap-old: ", c0, ", cap-new: ", c1, ", len-new: ", l1)
	reflect.Copy(s2, s)
	return s2
}
