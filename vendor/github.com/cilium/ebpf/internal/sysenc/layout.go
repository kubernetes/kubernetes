// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found at https://go.dev/LICENSE.

package sysenc

import (
	"reflect"
	"sync"
)

var hasUnexportedFieldsCache sync.Map // map[reflect.Type]bool

func hasUnexportedFields(typ reflect.Type) bool {
	switch typ.Kind() {
	case reflect.Slice, reflect.Array, reflect.Pointer:
		return hasUnexportedFields(typ.Elem())

	case reflect.Struct:
		if unexported, ok := hasUnexportedFieldsCache.Load(typ); ok {
			return unexported.(bool)
		}

		unexported := false
		for i, n := 0, typ.NumField(); i < n; i++ {
			field := typ.Field(i)
			// Package binary allows _ fields but always writes zeroes into them.
			if (!field.IsExported() && field.Name != "_") || hasUnexportedFields(field.Type) {
				unexported = true
				break
			}
		}

		hasUnexportedFieldsCache.Store(typ, unexported)
		return unexported

	default:
		// NB: It's not clear what this means for Chan and so on.
		return false
	}
}
