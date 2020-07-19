// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package mapsort provides sorted access to maps.
package mapsort

import (
	"sort"

	"google.golang.org/protobuf/reflect/protoreflect"
)

// Range iterates over every map entry in sorted key order,
// calling f for each key and value encountered.
func Range(mapv protoreflect.Map, keyKind protoreflect.Kind, f func(protoreflect.MapKey, protoreflect.Value) bool) {
	var keys []protoreflect.MapKey
	mapv.Range(func(key protoreflect.MapKey, _ protoreflect.Value) bool {
		keys = append(keys, key)
		return true
	})
	sort.Slice(keys, func(i, j int) bool {
		switch keyKind {
		case protoreflect.BoolKind:
			return !keys[i].Bool() && keys[j].Bool()
		case protoreflect.Int32Kind, protoreflect.Sint32Kind, protoreflect.Sfixed32Kind,
			protoreflect.Int64Kind, protoreflect.Sint64Kind, protoreflect.Sfixed64Kind:
			return keys[i].Int() < keys[j].Int()
		case protoreflect.Uint32Kind, protoreflect.Fixed32Kind,
			protoreflect.Uint64Kind, protoreflect.Fixed64Kind:
			return keys[i].Uint() < keys[j].Uint()
		case protoreflect.StringKind:
			return keys[i].String() < keys[j].String()
		default:
			panic("invalid kind: " + keyKind.String())
		}
	})
	for _, key := range keys {
		if !f(key, mapv.Get(key)) {
			break
		}
	}
}
