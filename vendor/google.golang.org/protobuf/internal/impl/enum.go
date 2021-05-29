// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package impl

import (
	"reflect"

	pref "google.golang.org/protobuf/reflect/protoreflect"
)

type EnumInfo struct {
	GoReflectType reflect.Type // int32 kind
	Desc          pref.EnumDescriptor
}

func (t *EnumInfo) New(n pref.EnumNumber) pref.Enum {
	return reflect.ValueOf(n).Convert(t.GoReflectType).Interface().(pref.Enum)
}
func (t *EnumInfo) Descriptor() pref.EnumDescriptor { return t.Desc }
