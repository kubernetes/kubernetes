// SPDX-FileCopyrightText: Copyright 2015-2025 go-swagger maintainers
// SPDX-License-Identifier: Apache-2.0

package conv

import (
	"strconv"
)

const baseDecimal = 10

// FormatInteger turns an integer type into a string.
func FormatInteger[T Signed](value T) string {
	return strconv.FormatInt(int64(value), baseDecimal)
}

// FormatUinteger turns an unsigned integer type into a string.
func FormatUinteger[T Unsigned](value T) string {
	return strconv.FormatUint(uint64(value), baseDecimal)
}

// FormatFloat turns a floating point numerical value into a string.
func FormatFloat[T Float](value T) string {
	return strconv.FormatFloat(float64(value), 'f', -1, bitsize(value))
}

// FormatBool turns a boolean into a string.
func FormatBool(value bool) string {
	return strconv.FormatBool(value)
}

// AppendInteger appends the decimal representation of an integer to a slice of bytes.
func AppendInteger[T Signed](dst []byte, value T) []byte {
	return strconv.AppendInt(dst, int64(value), baseDecimal)
}

// AppendUinteger appends the decimal representation of an unsigned integer to a slice of bytes.
func AppendUinteger[T Unsigned](dst []byte, value T) []byte {
	return strconv.AppendUint(dst, uint64(value), baseDecimal)
}

// AppendFloat appends the decimal representation of a floating point number to a slice of bytes.
func AppendFloat[T Float](dst []byte, value T) []byte {
	return strconv.AppendFloat(dst, float64(value), 'g', -1, bitsize(value))
}

// AppendBool appends the text representation of a boolean to a slice of bytes.
func AppendBool(dst []byte, value bool) []byte {
	return strconv.AppendBool(dst, value)
}
