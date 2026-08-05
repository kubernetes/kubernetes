// SPDX-FileCopyrightText: Copyright (c) 2015-2025 go-swagger maintainers
// SPDX-License-Identifier: Apache-2.0

package jsonpointer

import "reflect"

// JSONPointable is an interface for structs to implement, when they need to customize the json
// pointer process or want to avoid the use of reflection.
type JSONPointable interface {
	// JSONLookup returns a value pointed at this (unescaped) key.
	JSONLookup(key string) (any, error)
}

// JSONSetable is an interface for structs to implement, when they need to customize the json
// pointer process or want to avoid the use of reflection.
//
// # Handling of the RFC 6901 "-" token
//
// When a type implementing JSONSetable is the terminal parent of a [Pointer.Set] call, the library
// passes the raw reference token to JSONSet without interpretation.
//
// In particular, the RFC 6901 "-" token (which conventionally means "append" for arrays, per RFC
// 6902) is forwarded verbatim as the key argument.
//
// Implementations that model an array-like container are expected to give "-" the append semantics;
// implementations that do not should return an error wrapping [ErrDashToken] (or [ErrPointer]) for
// clarity.
//
// Implementations are responsible for any in-place mutation: the library does not attempt to rebind
// the result of JSONSet into a parent container.
type JSONSetable interface {
	// JSONSet sets the value pointed at the (unescaped) key.
	//
	// The key may be the RFC 6901 "-" token when the pointer targets a slice-like member; see the
	// interface documentation for details.
	JSONSet(key string, value any) error
}

// NameProvider knows how to resolve go struct fields into json names.
//
// The default provider is brought by
// [github.com/go-openapi/jsonpointer/jsonname.DefaultJSONNameProvider].
type NameProvider interface {
	// GetGoName gets the go name for a json property name
	GetGoName(subject any, name string) (string, bool)

	// GetGoNameForType gets the go name for a given type for a json property name
	GetGoNameForType(tpe reflect.Type, name string) (string, bool)
}
