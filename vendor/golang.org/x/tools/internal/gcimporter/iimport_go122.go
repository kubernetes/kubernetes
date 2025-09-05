// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.22 && !go1.24

package gcimporter

import (
	"go/token"
	"go/types"
	"unsafe"
)

// TODO(rfindley): delete this workaround once go1.24 is assured.

func init() {
	// Update markBlack so that it correctly sets the color
	// of imported TypeNames.
	//
	// See the doc comment for markBlack for details.

	type color uint32
	const (
		white color = iota
		black
		grey
	)
	type object struct {
		_      *types.Scope
		_      token.Pos
		_      *types.Package
		_      string
		_      types.Type
		_      uint32
		color_ color
		_      token.Pos
	}
	type typeName struct {
		object
	}

	// If the size of types.TypeName changes, this will fail to compile.
	const delta = int64(unsafe.Sizeof(typeName{})) - int64(unsafe.Sizeof(types.TypeName{}))
	var _ [-delta * delta]int

	markBlack = func(obj *types.TypeName) {
		type uP = unsafe.Pointer
		var ptr *typeName
		*(*uP)(uP(&ptr)) = uP(obj)
		ptr.color_ = black
	}
}
