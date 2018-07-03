// Copyright 2013 Dario Castañé. All rights reserved.
// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Based on src/pkg/reflect/deepequal.go from official
// golang's stdlib.

package mergo

import (
	"errors"
	"reflect"
)

// Errors reported by Mergo when it finds invalid arguments.
var (
	ErrNilArguments                = errors.New("src and dst must not be nil")
	ErrDifferentArgumentsTypes     = errors.New("src and dst must be of same type")
	ErrNotSupported                = errors.New("only structs and maps are supported")
	ErrExpectedMapAsDestination    = errors.New("dst was expected to be a map")
	ErrExpectedStructAsDestination = errors.New("dst was expected to be a struct")
)

// During deepMerge, must keep track of checks that are
// in progress.  The comparison algorithm assumes that all
// checks in progress are true when it reencounters them.
// Visited are stored in a map indexed by 17 * a1 + a2;
type visit struct {
	ptr  uintptr
	typ  reflect.Type
	next *visit
}

// From src/pkg/encoding/json/encode.go.
func isEmptyValue(v reflect.Value) bool {
	switch v.Kind() {
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
	case reflect.Interface, reflect.Ptr, reflect.Func:
		return v.IsNil()
	case reflect.Invalid:
		return true
	}
	return false
}

func resolveValues(dst, src interface{}) (vDst, vSrc reflect.Value, err error) {
	if dst == nil || src == nil {
		err = ErrNilArguments
		return
	}
	vDst = reflect.ValueOf(dst).Elem()
	if vDst.Kind() != reflect.Struct && vDst.Kind() != reflect.Map {
		err = ErrNotSupported
		return
	}
	vSrc = reflect.ValueOf(src)
	// We check if vSrc is a pointer to dereference it.
	if vSrc.Kind() == reflect.Ptr {
		vSrc = vSrc.Elem()
	}
	return
}

// Traverses recursively both values, assigning src's fields values to dst.
// The map argument tracks comparisons that have already been seen, which allows
// short circuiting on recursive types.
func deeper(dst, src reflect.Value, visited map[uintptr]*visit, depth int) (err error) {
	if dst.CanAddr() {
		addr := dst.UnsafeAddr()
		h := 17 * addr
		seen := visited[h]
		typ := dst.Type()
		for p := seen; p != nil; p = p.next {
			if p.ptr == addr && p.typ == typ {
				return nil
			}
		}
		// Remember, remember...
		visited[h] = &visit{addr, typ, seen}
	}
	return // TODO refactor
}
