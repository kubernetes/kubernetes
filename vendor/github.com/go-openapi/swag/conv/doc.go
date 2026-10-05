// SPDX-FileCopyrightText: Copyright 2015-2025 go-swagger maintainers
// SPDX-License-Identifier: Apache-2.0

// Package conv exposes utilities to convert types.
//
// The Convert and Format families of functions are essentially a shorthand to [strconv] functions,
// using the decimal representation of numbers.
//
// Features:
//
//   - from string representation to value ("Convert*") and reciprocally ("Format*")
//   - from pointer to value ([Value]) and reciprocally ([Pointer])
//   - from slice of values to slice of pointers ([PointerSlice]) and reciprocally ([ValueSlice])
//   - from map of values to map of pointers ([PointerMap]) and reciprocally ([ValueMap])
package conv
