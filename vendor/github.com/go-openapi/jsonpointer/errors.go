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
	ErrInvalidStart pointerError = `JSON pointer must be empty or start with a "` + pointerSeparator

	// ErrUnsupportedValueType indicates that a value of the wrong type is being set.
	ErrUnsupportedValueType pointerError = "only structs, pointers, maps and slices are supported for setting values"
)

func errNoKey(key string) error {
	return fmt.Errorf("object has no key %q: %w", key, ErrPointer)
}

func errOutOfBounds(length, idx int) error {
	return fmt.Errorf("index out of bounds array[0,%d] index '%d': %w", length-1, idx, ErrPointer)
}

func errInvalidReference(token string) error {
	return fmt.Errorf("invalid token reference %q: %w", token, ErrPointer)
}
