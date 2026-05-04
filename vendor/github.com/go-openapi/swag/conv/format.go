// SPDX-FileCopyrightText: Copyright 2015-2025 go-swagger maintainers
// SPDX-License-Identifier: Apache-2.0

package conv

import (
	"strconv"
)

// FormatInteger turns an integer type into a string.
func FormatInteger[T Signed](value T) string {
	return strconv.FormatInt(int64(value), 10)
}

// FormatUinteger turns an unsigned integer type into a string.
func FormatUinteger[T Unsigned](value T) string {
	return strconv.FormatUint(uint64(value), 10)
}

// FormatFloat turns a floating point numerical value into a string.
func FormatFloat[T Float](value T) string {
	return strconv.FormatFloat(float64(value), 'f', -1, bitsize(value))
}

// FormatBool turns a boolean into a string.
func FormatBool(value bool) string {
	return strconv.FormatBool(value)
}
