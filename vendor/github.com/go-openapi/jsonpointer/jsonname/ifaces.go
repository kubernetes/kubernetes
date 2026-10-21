// SPDX-FileCopyrightText: Copyright 2015-2025 go-swagger maintainers
// SPDX-License-Identifier: Apache-2.0

package jsonname

import "reflect"

// providerIface is an unexported compile-time contract that every name provider in this package is
// expected to satisfy.
//
// It mirrors the interface declared by the main consumer of this module:
// [github.com/go-openapi/jsonpointer.NameProvider].
type providerIface interface {
	GetGoName(subject any, name string) (string, bool)
	GetGoNameForType(tpe reflect.Type, name string) (string, bool)
}
