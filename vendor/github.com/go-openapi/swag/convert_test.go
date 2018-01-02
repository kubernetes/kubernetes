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
	"testing"

	"github.com/stretchr/testify/assert"
)

// These are really dumb tests

func TestConvertBool(t *testing.T) {
	for k := range evaluatesAsTrue {
		r, err := ConvertBool(k)
		if assert.NoError(t, err) {
			assert.True(t, r)
		}
	}
	for _, k := range []string{"a", "", "0", "false", "unchecked"} {
		r, err := ConvertBool(k)
		if assert.NoError(t, err) {
			assert.False(t, r)
		}
	}
}

func TestConvertFloat32(t *testing.T) {
	validFloats := []float32{1.0, -1, math.MaxFloat32, math.SmallestNonzeroFloat32, 0, 5.494430303}
	invalidFloats := []string{"a", strconv.FormatFloat(math.MaxFloat64, 'f', -1, 64), "true"}

	for _, f := range validFloats {
		c, err := ConvertFloat32(FormatFloat32(f))
		if assert.NoError(t, err) {
			assert.EqualValues(t, f, c)
		}
	}
	for _, f := range invalidFloats {
		_, err := ConvertFloat32(f)
		assert.Error(t, err, "expected '"+f+"' to generate an error")
	}
}

func TestConvertFloat64(t *testing.T) {
	validFloats := []float64{1.0, -1, float64(math.MaxFloat32), float64(math.SmallestNonzeroFloat32), math.MaxFloat64, math.SmallestNonzeroFloat64, 0, 5.494430303}
	invalidFloats := []string{"a", "true"}

	for _, f := range validFloats {
		c, err := ConvertFloat64(FormatFloat64(f))
		if assert.NoError(t, err) {
			assert.EqualValues(t, f, c)
		}
	}
	for _, f := range invalidFloats {
		_, err := ConvertFloat64(f)
		assert.Error(t, err, "expected '"+f+"' to generate an error")
	}
}

func TestConvertInt8(t *testing.T) {
	validInts := []int8{0, 1, -1, math.MaxInt8, math.MinInt8}
	invalidInts := []string{"1.233", "a", "false", strconv.Itoa(int(math.MaxInt64))}

	for _, f := range validInts {
		c, err := ConvertInt8(FormatInt8(f))
		if assert.NoError(t, err) {
			assert.EqualValues(t, f, c)
		}
	}
	for _, f := range invalidInts {
		_, err := ConvertInt8(f)
		assert.Error(t, err, "expected '"+f+"' to generate an error")
	}
}

func TestConvertInt16(t *testing.T) {
	validInts := []int16{0, 1, -1, math.MaxInt8, math.MinInt8, math.MaxInt16, math.MinInt16}
	invalidInts := []string{"1.233", "a", "false", strconv.Itoa(int(math.MaxInt64))}

	for _, f := range validInts {
		c, err := ConvertInt16(FormatInt16(f))
		if assert.NoError(t, err) {
			assert.EqualValues(t, f, c)
		}
	}
	for _, f := range invalidInts {
		_, err := ConvertInt16(f)
		assert.Error(t, err, "expected '"+f+"' to generate an error")
	}
}

func TestConvertInt32(t *testing.T) {
	validInts := []int32{0, 1, -1, math.MaxInt8, math.MinInt8, math.MaxInt16, math.MinInt16, math.MinInt32, math.MaxInt32}
	invalidInts := []string{"1.233", "a", "false", strconv.Itoa(int(math.MaxInt64))}

	for _, f := range validInts {
		c, err := ConvertInt32(FormatInt32(f))
		if assert.NoError(t, err) {
			assert.EqualValues(t, f, c)
		}
	}
	for _, f := range invalidInts {
		_, err := ConvertInt32(f)
		assert.Error(t, err, "expected '"+f+"' to generate an error")
	}
}

func TestConvertInt64(t *testing.T) {
	validInts := []int64{0, 1, -1, math.MaxInt8, math.MinInt8, math.MaxInt16, math.MinInt16, math.MinInt32, math.MaxInt32, math.MaxInt64, math.MinInt64}
	invalidInts := []string{"1.233", "a", "false"}

	for _, f := range validInts {
		c, err := ConvertInt64(FormatInt64(f))
		if assert.NoError(t, err) {
			assert.EqualValues(t, f, c)
		}
	}
	for _, f := range invalidInts {
		_, err := ConvertInt64(f)
		assert.Error(t, err, "expected '"+f+"' to generate an error")
	}
}

func TestConvertUint8(t *testing.T) {
	validInts := []uint8{0, 1, math.MaxUint8}
	invalidInts := []string{"1.233", "a", "false", strconv.FormatUint(math.MaxUint64, 10)}

	for _, f := range validInts {
		c, err := ConvertUint8(FormatUint8(f))
		if assert.NoError(t, err) {
			assert.EqualValues(t, f, c)
		}
	}
	for _, f := range invalidInts {
		_, err := ConvertUint8(f)
		assert.Error(t, err, "expected '"+f+"' to generate an error")
	}
}

func TestConvertUint16(t *testing.T) {
	validUints := []uint16{0, 1, math.MaxUint8, math.MaxUint16}
	invalidUints := []string{"1.233", "a", "false", strconv.FormatUint(math.MaxUint64, 10)}

	for _, f := range validUints {
		c, err := ConvertUint16(FormatUint16(f))
		if assert.NoError(t, err) {
			assert.EqualValues(t, f, c)
		}
	}
	for _, f := range invalidUints {
		_, err := ConvertUint16(f)
		assert.Error(t, err, "expected '"+f+"' to generate an error")
	}
}

func TestConvertUint32(t *testing.T) {
	validUints := []uint32{0, 1, math.MaxUint8, math.MaxUint16, math.MaxUint32}
	invalidUints := []string{"1.233", "a", "false", strconv.FormatUint(math.MaxUint64, 10)}

	for _, f := range validUints {
		c, err := ConvertUint32(FormatUint32(f))
		if assert.NoError(t, err) {
			assert.EqualValues(t, f, c)
		}
	}
	for _, f := range invalidUints {
		_, err := ConvertUint32(f)
		assert.Error(t, err, "expected '"+f+"' to generate an error")
	}
}

func TestConvertUint64(t *testing.T) {
	validUints := []uint64{0, 1, math.MaxUint8, math.MaxUint16, math.MaxUint32, math.MaxUint64}
	invalidUints := []string{"1.233", "a", "false"}

	for _, f := range validUints {
		c, err := ConvertUint64(FormatUint64(f))
		if assert.NoError(t, err) {
			assert.EqualValues(t, f, c)
		}
	}
	for _, f := range invalidUints {
		_, err := ConvertUint64(f)
		assert.Error(t, err, "expected '"+f+"' to generate an error")
	}
}

func TestIsFloat64AJSONInteger(t *testing.T) {
	assert.False(t, IsFloat64AJSONInteger(math.Inf(1)))
	assert.False(t, IsFloat64AJSONInteger(maxJSONFloat+1))

	assert.False(t, IsFloat64AJSONInteger(minJSONFloat-1))
	assert.True(t, IsFloat64AJSONInteger(1.0))
	assert.True(t, IsFloat64AJSONInteger(maxJSONFloat))
	assert.True(t, IsFloat64AJSONInteger(minJSONFloat))
}

func TestFormatBool(t *testing.T) {
	assert.Equal(t, "true", FormatBool(true))
	assert.Equal(t, "false", FormatBool(false))
}
