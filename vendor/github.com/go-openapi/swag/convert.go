// Copyright 2015 go-swagger maintainers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package swag

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

// IsFloat64AJSONInteger allow for integers [-2^53, 2^53-1] inclusive
func IsFloat64AJSONInteger(f float64) bool {
	if math.IsNaN(f) || math.IsInf(f, 0) || f < minJSONFloat || f > maxJSONFloat {
		return false
	}
	fa := math.Abs(f)
	g := float64(uint64(f))
	ga := math.Abs(g)

	diff := math.Abs(f - g)

	// more info: https://floating-point-gui.de/errors/comparison/#look-out-for-edge-cases
	switch {
	case f == g: // best case
		return true
	case f == float64(int64(f)) || f == float64(uint64(f)): // optimistic case
		return true
	case f == 0 || g == 0 || diff < math.SmallestNonzeroFloat64: // very close to 0 values
		return diff < (epsilon * math.SmallestNonzeroFloat64)
	}
	// check the relative error
	return diff/math.Min(fa+ga, math.MaxFloat64) < epsilon
}

var evaluatesAsTrue map[string]struct{}

func init() {
	evaluatesAsTrue = map[string]struct{}{
		"true":     {},
		"1":        {},
		"yes":      {},
		"ok":       {},
		"y":        {},
		"on":       {},
		"selected": {},
		"checked":  {},
		"t":        {},
		"enabled":  {},
	}
}

// ConvertBool turn a string into a boolean
func ConvertBool(str string) (bool, error) {
	_, ok := evaluatesAsTrue[strings.ToLower(str)]
	return ok, nil
}

// ConvertFloat32 turn a string into a float32
func ConvertFloat32(str string) (float32, error) {
	f, err := strconv.ParseFloat(str, 32)
	if err != nil {
		return 0, err
	}
	return float32(f), nil
}

// ConvertFloat64 turn a string into a float64
func ConvertFloat64(str string) (float64, error) {
	return strconv.ParseFloat(str, 64)
}

// ConvertInt8 turn a string into int8 boolean
func ConvertInt8(str string) (int8, error) {
	i, err := strconv.ParseInt(str, 10, 8)
	if err != nil {
		return 0, err
	}
	return int8(i), nil
}

// ConvertInt16 turn a string into a int16
func ConvertInt16(str string) (int16, error) {
	i, err := strconv.ParseInt(str, 10, 16)
	if err != nil {
		return 0, err
	}
	return int16(i), nil
}

// ConvertInt32 turn a string into a int32
func ConvertInt32(str string) (int32, error) {
	i, err := strconv.ParseInt(str, 10, 32)
	if err != nil {
		return 0, err
	}
	return int32(i), nil
}

// ConvertInt64 turn a string into a int64
func ConvertInt64(str string) (int64, error) {
	return strconv.ParseInt(str, 10, 64)
}

// ConvertUint8 turn a string into a uint8
func ConvertUint8(str string) (uint8, error) {
	i, err := strconv.ParseUint(str, 10, 8)
	if err != nil {
		return 0, err
	}
	return uint8(i), nil
}

// ConvertUint16 turn a string into a uint16
func ConvertUint16(str string) (uint16, error) {
	i, err := strconv.ParseUint(str, 10, 16)
	if err != nil {
		return 0, err
	}
	return uint16(i), nil
}

// ConvertUint32 turn a string into a uint32
func ConvertUint32(str string) (uint32, error) {
	i, err := strconv.ParseUint(str, 10, 32)
	if err != nil {
		return 0, err
	}
	return uint32(i), nil
}

// ConvertUint64 turn a string into a uint64
func ConvertUint64(str string) (uint64, error) {
	return strconv.ParseUint(str, 10, 64)
}

// FormatBool turns a boolean into a string
func FormatBool(value bool) string {
	return strconv.FormatBool(value)
}

// FormatFloat32 turns a float32 into a string
func FormatFloat32(value float32) string {
	return strconv.FormatFloat(float64(value), 'f', -1, 32)
}

// FormatFloat64 turns a float64 into a string
func FormatFloat64(value float64) string {
	return strconv.FormatFloat(value, 'f', -1, 64)
}

// FormatInt8 turns an int8 into a string
func FormatInt8(value int8) string {
	return strconv.FormatInt(int64(value), 10)
}

// FormatInt16 turns an int16 into a string
func FormatInt16(value int16) string {
	return strconv.FormatInt(int64(value), 10)
}

// FormatInt32 turns an int32 into a string
func FormatInt32(value int32) string {
	return strconv.Itoa(int(value))
}

// FormatInt64 turns an int64 into a string
func FormatInt64(value int64) string {
	return strconv.FormatInt(value, 10)
}

// FormatUint8 turns an uint8 into a string
func FormatUint8(value uint8) string {
	return strconv.FormatUint(uint64(value), 10)
}

// FormatUint16 turns an uint16 into a string
func FormatUint16(value uint16) string {
	return strconv.FormatUint(uint64(value), 10)
}

// FormatUint32 turns an uint32 into a string
func FormatUint32(value uint32) string {
	return strconv.FormatUint(uint64(value), 10)
}

// FormatUint64 turns an uint64 into a string
func FormatUint64(value uint64) string {
	return strconv.FormatUint(value, 10)
}
