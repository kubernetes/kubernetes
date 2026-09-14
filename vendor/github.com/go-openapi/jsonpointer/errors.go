// SPDX-FileCopyrightText: Copyright (c) 2015-2025 go-swagger maintainers
// SPDX-License-Identifier: Apache-2.0

package jsonpointer

import "fmt"

type pointerError string

func (e pointerError) Error() string {
	return string(e)
}

const (
	// ErrPointer is a sentinel error raised by all errors from this package.
	ErrPointer pointerError = "JSON pointer error"

	// ErrInvalidStart states that a JSON pointer must start with a separator ("/").
	ErrInvalidStart pointerError = `JSON pointer must be empty or start with a "` + pointerSeparator + `"`

	// ErrUnsupportedValueType indicates that a value of the wrong type is being set.
	ErrUnsupportedValueType pointerError = "only structs, pointers, maps and slices are supported for setting values"

	// ErrDashToken indicates use of the RFC 6901 "-" reference token in a context where it cannot be
	// resolved.
	//
	// Per RFC 6901 §4 the "-" token refers to the (nonexistent) element after the last array element.
	// It may only be used as the terminal token of a [Pointer.Set] against a slice, where it means
	// "append".
	//
	// Any other use (get, offset, intermediate traversal, non-slice target) is an error condition that
	// wraps this sentinel.
	ErrDashToken pointerError = `the "-" array token cannot be resolved here` //nolint:gosec // G101 false positive: this is a JSON Pointer reference token, not a credential.
)

const dashToken = "-"

func errNoKey(key string) error {
	return fmt.Errorf("object has no key %q: %w", key, ErrPointer)
}

func errOutOfBounds(length, idx int) error {
	return fmt.Errorf("index out of bounds array[0,%d] index '%d': %w", length-1, idx, ErrPointer)
}

func errInvalidReference(token string) error {
	return fmt.Errorf("invalid token reference %q: %w", token, ErrPointer)
}

func errDashOnGet() error {
	return fmt.Errorf("cannot resolve %q token on get: %w: %w", dashToken, ErrDashToken, ErrPointer)
}

func errDashIntermediate() error {
	return fmt.Errorf("the %q token may only appear as the terminal token of a pointer: %w: %w", dashToken, ErrDashToken, ErrPointer)
}

func errDashOnOffset() error {
	return fmt.Errorf("cannot compute offset for %q token (nonexistent element): %w: %w", dashToken, ErrDashToken, ErrPointer)
}
