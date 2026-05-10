// SPDX-FileCopyrightText: Copyright 2015-2025 go-swagger maintainers
// SPDX-License-Identifier: Apache-2.0

package conv

import (
	"math"
	"strconv"
	"strings"
)

// same as ECMA Number.MAX_SAFE_INTEGER and Number.MIN_SAFE_INTEGER
const (
	maxJSONFloat         = float64(1<<53 - 1)  // 9007199254740991.0 	 	 2^53 - 1
	minJSONFloat         = -float64(1<<53 - 1) //-9007199254740991.0	-2^53 - 1
	epsilon      float64 = 1e-9
)

// IsFloat64AJSONInteger allows for integers [-2^53, 2^53-1] inclusive.
func IsFloat64AJSONInteger(f float64) bool {
	if math.IsNaN(f) || math.IsInf(f, 0) || f < minJSONFloat || f > maxJSONFloat {
		return false
	}
	rounded := math.Round(f)
	if f == rounded {
		return true
	}
	if rounded == 0 { // f = 0.0 exited above
		return false
	}

	diff := math.Abs(f - rounded)
	if diff == 0 {
		return true
	}

	// relative error Abs{f - Round(f)) / Round(f)} < ε ; Round(f)
	return diff < epsilon*math.Abs(rounded)
}

// ConvertFloat turns a string into a float numerical value.
func ConvertFloat[T Float](str string) (T, error) {
	var v T
	f, err := strconv.ParseFloat(str, bitsize(v))
	if err != nil {
		return 0, err
	}

	return T(f), nil
}

// ConvertInteger turns a string into a signed integer.
func ConvertInteger[T Signed](str string) (T, error) {
	var v T
	f, err := strconv.ParseInt(str, 10, bitsize(v))
	if err != nil {
		return 0, err
	}

	return T(f), nil
}

// ConvertUinteger turns a string into an unsigned integer.
func ConvertUinteger[T Unsigned](str string) (T, error) {
	var v T
	f, err := strconv.ParseUint(str, 10, bitsize(v))
	if err != nil {
		return 0, err
	}

	return T(f), nil
}

// ConvertBool turns a string into a boolean.
//
// It supports a few more "true" strings than [strconv.ParseBool]:
//
//   - it is not case sensitive ("trUe" or "FalsE" work)
//   - "ok", "yes", "y", "on", "selected", "checked", "enabled" are all true
//   - everything that is not true is false: there is never an actual error returned
func ConvertBool(str string) (bool, error) {
	switch strings.ToLower(str) {
	case "true",
		"1",
		"yes",
		"ok",
		"y",
		"on",
		"selected",
		"checked",
		"t",
		"enabled":
		return true, nil
	default:
		return false, nil
	}
}

// ConvertFloat32 turns a string into a float32.
func ConvertFloat32(str string) (float32, error) { return ConvertFloat[float32](str) }

// ConvertFloat64 turns a string into a float64
func ConvertFloat64(str string) (float64, error) { return ConvertFloat[float64](str) }

// ConvertInt8 turns a string into an int8
func ConvertInt8(str string) (int8, error) { return ConvertInteger[int8](str) }

// ConvertInt16 turns a string into an int16
func ConvertInt16(str string) (int16, error) {
	i, err := strconv.ParseInt(str, 10, 16)
	if err != nil {
		return 0, err
	}
	return int16(i), nil
}

// ConvertInt32 turns a string into an int32
func ConvertInt32(str string) (int32, error) {
	i, err := strconv.ParseInt(str, 10, 32)
	if err != nil {
		return 0, err
	}
	return int32(i), nil
}

// ConvertInt64 turns a string into an int64
func ConvertInt64(str string) (int64, error) {
	return strconv.ParseInt(str, 10, 64)
}

// ConvertUint8 turns a string into an uint8
func ConvertUint8(str string) (uint8, error) {
	i, err := strconv.ParseUint(str, 10, 8)
	if err != nil {
		return 0, err
	}
	return uint8(i), nil
}

// ConvertUint16 turns a string into an uint16
func ConvertUint16(str string) (uint16, error) {
	i, err := strconv.ParseUint(str, 10, 16)
	if err != nil {
		return 0, err
	}
	return uint16(i), nil
}

// ConvertUint32 turns a string into an uint32
func ConvertUint32(str string) (uint32, error) {
	i, err := strconv.ParseUint(str, 10, 32)
	if err != nil {
		return 0, err
	}
	return uint32(i), nil
}

// ConvertUint64 turns a string into an uint64
func ConvertUint64(str string) (uint64, error) {
	return strconv.ParseUint(str, 10, 64)
}
