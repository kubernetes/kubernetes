// SPDX-FileCopyrightText: Copyright 2015-2025 go-swagger maintainers
// SPDX-License-Identifier: Apache-2.0

package conv

import "unsafe"

// bitsize returns the size in bits of a type.
//
// NOTE: [unsafe.SizeOf] simply returns the size in bytes of the value.
// For primitive types T, the generic stencil is precompiled and this value
// is resolved at compile time, resulting in an immediate call to [strconv.ParseFloat].
//
// We may leave up to the go compiler to simplify this function into a
// constant value, which happens in practice at least for primitive types
// (e.g. numerical types).
func bitsize[T Numerical](value T) int {
	const bitsPerByte = 8
	return int(unsafe.Sizeof(value)) * bitsPerByte
}
