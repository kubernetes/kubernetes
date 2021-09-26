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

package validate

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"k8s.io/kube-openapi/pkg/validation/errors"
	"k8s.io/kube-openapi/pkg/validation/strfmt"
)

func TestValues_ValidateIntEnum(t *testing.T) {
	enumValues := []interface{}{1, 2, 3}

	err := Enum("test", "body", int64(5), enumValues)
	assert.NotNil(t, err)
	err2 := Enum("test", "body", int64(1), enumValues)
	assert.Nil(t, err2)
}

func TestValues_ValidateEnum(t *testing.T) {
	enumValues := []string{"aa", "bb", "cc"}

	err := Enum("test", "body", "a", enumValues)
	assert.Error(t, err)
	err = Enum("test", "body", "bb", enumValues)
	assert.Nil(t, err)
}

// Check edge cases in Enum
func TestValues_Enum_EdgeCases(t *testing.T) {
	enumValues := "aa, bb, cc"

	err := Enum("test", "body", int64(1), enumValues)
	// No validation occurs: enumValues is not a slice
	assert.Nil(t, err)

	// TODO(TEST): edge case: value is not a concrete type
	// It's really a go internals challenge
	// to figure a test case to demonstrate
	// this case must be checked (!!)
}

func TestValues_ValidateUniqueItems(t *testing.T) {
	var err error

	itemsNonUnique := []interface{}{
		[]int32{1, 2, 3, 4, 4, 5},
		[]string{"aa", "bb", "cc", "cc", "dd"},
	}
	for _, v := range itemsNonUnique {
		err = UniqueItems("test", "body", v)
		assert.Error(t, err)
	}

	itemsUnique := []interface{}{
		[]int32{1, 2, 3},
		"I'm a string",
		map[string]int{
			"aaa": 1111,
			"b":   2,
			"ccc": 333,
		},
		nil,
	}
	for _, v := range itemsUnique {
		err = UniqueItems("test", "body", v)
		assert.Nil(t, err)
	}
}

func TestValues_ValidateMinLength(t *testing.T) {
	var minLength int64 = 5
	err := MinLength("test", "body", "aa", minLength)
	assert.Error(t, err)
	err = MinLength("test", "body", "aaaaa", minLength)
	assert.Nil(t, err)
}

func TestValues_ValidateMaxLength(t *testing.T) {
	var maxLength int64 = 5
	err := MaxLength("test", "body", "bbbbbb", maxLength)
	assert.Error(t, err)
	err = MaxLength("test", "body", "aa", maxLength)
	assert.Nil(t, err)
}

func TestValues_ValidateRequired(t *testing.T) {
	var err error
	path := "test"
	in := "body"

	RequiredFail := []interface{}{
		"",
		0,
		nil,
	}

	for _, v := range RequiredFail {
		err = Required(path, in, v)
		assert.Error(t, err)
	}

	RequiredSuccess := []interface{}{
		" ",
		"bla-bla-bla",
		2,
		[]interface{}{21, []int{}, "testString"},
	}

	for _, v := range RequiredSuccess {
		err = Required(path, in, v)
		assert.Nil(t, err)
	}

}

func TestValuMultipleOf(t *testing.T) {

	// positive

	err := MultipleOf("test", "body", 9, 3)
	assert.Nil(t, err)

	err = MultipleOf("test", "body", 9.3, 3.1)
	assert.Nil(t, err)

	err = MultipleOf("test", "body", 9.1, 0.1)
	assert.Nil(t, err)

	err = MultipleOf("test", "body", 3, 0.3)
	assert.Nil(t, err)

	err = MultipleOf("test", "body", 6, 0.3)
	assert.Nil(t, err)

	err = MultipleOf("test", "body", 1, 0.25)
	assert.Nil(t, err)

	err = MultipleOf("test", "body", 8, 0.2)
	assert.Nil(t, err)

	// negative

	err = MultipleOf("test", "body", 3, 0.4)
	assert.Error(t, err)

	err = MultipleOf("test", "body", 9.1, 0.2)
	assert.Error(t, err)

	err = MultipleOf("test", "body", 9.34, 0.1)
	assert.Error(t, err)

	// error on negative factor
	err = MultipleOf("test", "body", 9.34, -0.1)
	assert.Error(t, err)
}

// Test edge case for Pattern (in regular spec, no invalid regexp should reach there)
func TestValues_Pattern_Edgecases(t *testing.T) {
	var err *errors.Validation
	err = Pattern("path", "in", "pick-a-boo", `.*-[a-z]-.*`)
	assert.Nil(t, err)

	// Invalid regexp
	err = Pattern("path", "in", "pick-a-boo", `.*-[a(-z]-^).*`)
	if assert.NotNil(t, err) {
		assert.Equal(t, int(err.Code()), int(errors.PatternFailCode))
		assert.Contains(t, err.Error(), "pattern is invalid")
	}

	// Valid regexp, invalid pattern
	err = Pattern("path", "in", "pick-8-boo", `.*-[a-z]-.*`)
	if assert.NotNil(t, err) {
		assert.Equal(t, int(err.Code()), int(errors.PatternFailCode))
		assert.NotContains(t, err.Error(), "pattern is invalid")
		assert.Contains(t, err.Error(), "should match")
	}
}

// Test edge cases in FormatOf
// not easily tested with full specs
func TestValues_FormatOf_EdgeCases(t *testing.T) {
	var err *errors.Validation

	err = FormatOf("path", "in", "bugz", "", nil)
	if assert.NotNil(t, err) {
		assert.Equal(t, int(err.Code()), int(errors.InvalidTypeCode))
		assert.Contains(t, err.Error(), "bugz is an invalid type name")
	}

	err = FormatOf("path", "in", "bugz", "", strfmt.Default)
	if assert.NotNil(t, err) {
		assert.Equal(t, int(err.Code()), int(errors.InvalidTypeCode))
		assert.Contains(t, err.Error(), "bugz is an invalid type name")
	}
}

// Test edge cases in MaximumNativeType
// not easily exercised with full specs
func TestValues_MaximumNative(t *testing.T) {
	assert.Nil(t, MaximumNativeType("path", "in", int(5), 10, false))
	assert.Nil(t, MaximumNativeType("path", "in", uint(5), 10, true))
	assert.Nil(t, MaximumNativeType("path", "in", int8(5), 10, true))
	assert.Nil(t, MaximumNativeType("path", "in", uint8(5), 10, true))
	assert.Nil(t, MaximumNativeType("path", "in", int16(5), 10, true))
	assert.Nil(t, MaximumNativeType("path", "in", uint16(5), 10, true))
	assert.Nil(t, MaximumNativeType("path", "in", int32(5), 10, true))
	assert.Nil(t, MaximumNativeType("path", "in", uint32(5), 10, true))
	assert.Nil(t, MaximumNativeType("path", "in", int64(5), 10, true))
	assert.Nil(t, MaximumNativeType("path", "in", uint64(5), 10, true))
	assert.Nil(t, MaximumNativeType("path", "in", float32(5.5), 10, true))
	assert.Nil(t, MaximumNativeType("path", "in", float64(5.5), 10, true))

	var err *errors.Validation

	err = MaximumNativeType("path", "in", int32(10), 10, true)
	if assert.NotNil(t, err) {
		code := int(err.Code())
		assert.Equal(t, code, errors.MaxFailCode)
	}

	err = MaximumNativeType("path", "in", uint(10), 10, true)
	if assert.NotNil(t, err) {
		code := int(err.Code())
		assert.Equal(t, code, errors.MaxFailCode)
	}

	err = MaximumNativeType("path", "in", int64(12), 10, false)
	if assert.NotNil(t, err) {
		code := int(err.Code())
		assert.Equal(t, code, errors.MaxFailCode)
	}

	err = MaximumNativeType("path", "in", float32(12.6), 10, false)
	if assert.NotNil(t, err) {
		code := int(err.Code())
		assert.Equal(t, code, int(errors.MaxFailCode))
	}

	err = MaximumNativeType("path", "in", float64(12.6), 10, false)
	if assert.NotNil(t, err) {
		code := int(err.Code())
		assert.Equal(t, code, int(errors.MaxFailCode))
	}

	err = MaximumNativeType("path", "in", uint(5), -10, true)
	if assert.NotNil(t, err) {
		code := int(err.Code())
		assert.Equal(t, code, int(errors.MaxFailCode))
	}
}

// Test edge cases in MinimumNativeType
// not easily exercised with full specs
func TestValues_MinimumNative(t *testing.T) {
	assert.Nil(t, MinimumNativeType("path", "in", int(5), 0, false))
	assert.Nil(t, MinimumNativeType("path", "in", uint(5), 0, true))
	assert.Nil(t, MinimumNativeType("path", "in", int8(5), 0, true))
	assert.Nil(t, MinimumNativeType("path", "in", uint8(5), 0, true))
	assert.Nil(t, MinimumNativeType("path", "in", int16(5), 0, true))
	assert.Nil(t, MinimumNativeType("path", "in", uint16(5), 0, true))
	assert.Nil(t, MinimumNativeType("path", "in", int32(5), 0, true))
	assert.Nil(t, MinimumNativeType("path", "in", uint32(5), 0, true))
	assert.Nil(t, MinimumNativeType("path", "in", int64(5), 0, true))
	assert.Nil(t, MinimumNativeType("path", "in", uint64(5), 0, true))
	assert.Nil(t, MinimumNativeType("path", "in", float32(5.5), 0, true))
	assert.Nil(t, MinimumNativeType("path", "in", float64(5.5), 0, true))

	var err *errors.Validation

	err = MinimumNativeType("path", "in", uint(10), 10, true)
	if assert.NotNil(t, err) {
		code := int(err.Code())
		assert.Equal(t, code, int(errors.MinFailCode))
	}

	err = MinimumNativeType("path", "in", uint(10), 10, true)
	if assert.NotNil(t, err) {
		code := int(err.Code())
		assert.Equal(t, code, int(errors.MinFailCode))
	}

	err = MinimumNativeType("path", "in", int64(8), 10, false)
	if assert.NotNil(t, err) {
		code := int(err.Code())
		assert.Equal(t, code, int(errors.MinFailCode))
	}

	err = MinimumNativeType("path", "in", float32(12.6), 20, false)
	if assert.NotNil(t, err) {
		code := int(err.Code())
		assert.Equal(t, code, int(errors.MinFailCode))
	}

	err = MinimumNativeType("path", "in", float64(12.6), 20, false)
	if assert.NotNil(t, err) {
		code := int(err.Code())
		assert.Equal(t, code, int(errors.MinFailCode))
	}

	err = MinimumNativeType("path", "in", uint(5), -10, true)
	assert.Nil(t, err)
}

// Test edge cases in MaximumNativeType
// not easily exercised with full specs
func TestValues_MultipleOfNative(t *testing.T) {
	assert.Nil(t, MultipleOfNativeType("path", "in", int(5), 1))
	assert.Nil(t, MultipleOfNativeType("path", "in", uint(5), 1))
	assert.Nil(t, MultipleOfNativeType("path", "in", int8(5), 1))
	assert.Nil(t, MultipleOfNativeType("path", "in", uint8(5), 1))
	assert.Nil(t, MultipleOfNativeType("path", "in", int16(5), 1))
	assert.Nil(t, MultipleOfNativeType("path", "in", uint16(5), 1))
	assert.Nil(t, MultipleOfNativeType("path", "in", int32(5), 1))
	assert.Nil(t, MultipleOfNativeType("path", "in", uint32(5), 1))
	assert.Nil(t, MultipleOfNativeType("path", "in", int64(5), 1))
	assert.Nil(t, MultipleOfNativeType("path", "in", uint64(5), 1))

	var err *errors.Validation

	err = MultipleOfNativeType("path", "in", int64(5), -1)
	if assert.NotNil(t, err) {
		code := int(err.Code())
		assert.Equal(t, code, int(errors.MultipleOfMustBePositiveCode))
	}

	err = MultipleOfNativeType("path", "in", int64(11), 5)
	if assert.NotNil(t, err) {
		code := int(err.Code())
		assert.Equal(t, code, int(errors.MultipleOfFailCode))
	}

	err = MultipleOfNativeType("path", "in", uint64(11), 5)
	if assert.NotNil(t, err) {
		code := int(err.Code())
		assert.Equal(t, code, int(errors.MultipleOfFailCode))
	}
}

// Test edge cases in IsValueValidAgainstRange
// not easily exercised with full specs
func TestValues_IsValueValidAgainstRange(t *testing.T) {
	var err error

	// We did not simulate these formats in full specs
	err = IsValueValidAgainstRange(float32(123.45), "number", "float32", "prefix", "path")
	assert.NoError(t, err)

	err = IsValueValidAgainstRange(float64(123.45), "number", "float32", "prefix", "path")
	assert.NoError(t, err)

	err = IsValueValidAgainstRange(int64(123), "number", "float", "prefix", "path")
	assert.NoError(t, err)

	err = IsValueValidAgainstRange(int64(123), "integer", "", "prefix", "path")
	assert.NoError(t, err)

	err = IsValueValidAgainstRange(int64(123), "integer", "int64", "prefix", "path")
	assert.NoError(t, err)

	err = IsValueValidAgainstRange(int64(123), "integer", "uint64", "prefix", "path")
	assert.NoError(t, err)

	// Error case (do not occur in normal course of a validation)
	err = IsValueValidAgainstRange(float64(math.MaxFloat64), "integer", "", "prefix", "path")
	if assert.Error(t, err) {
		assert.Contains(t, err.Error(), "must be of type integer (default format)")
	}

	// Checking a few limits
	err = IsValueValidAgainstRange("123", "number", "", "prefix", "path")
	if assert.Error(t, err) {
		assert.Contains(t, err.Error(), "called with invalid (non numeric) val type")
	}

	err = IsValueValidAgainstRange(int64(2147483647), "integer", "int32", "prefix", "path")
	assert.NoError(t, err)

	err = IsValueValidAgainstRange(int64(2147483647), "integer", "uint32", "prefix", "path")
	assert.NoError(t, err)
}
