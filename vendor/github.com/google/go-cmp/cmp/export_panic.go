// Copyright 2017, The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE.md file.

// +build purego

package cmp

import "reflect"

const supportExporters = false

func retrieveUnexportedField(reflect.Value, reflect.StructField) reflect.Value {
	panic("no support for forcibly accessing unexported fields")
}
