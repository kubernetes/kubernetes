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
	"testing"

	"github.com/stretchr/testify/assert"
)

func integerFactory(base int) []interface{} {
	return []interface{}{
		base,
		int8(base),
		int16(base),
		int32(base),
		int64(base),
		uint(base),
		uint8(base),
		uint16(base),
		uint32(base),
		uint64(base),
		float32(base),
		float64(base),
	}
}

// Test cases in private method asInt64()
func TestHelpers_asInt64(t *testing.T) {
	for _, v := range integerFactory(3) {
		assert.Equal(t, int64(3), valueHelp.asInt64(v))
	}

	// Non numeric
	if assert.NotPanics(t, func() {
		valueHelp.asInt64("123")
	}) {
		assert.Equal(t, valueHelp.asInt64("123"), (int64)(0))
	}
}

// Test cases in private method asUint64()
func TestHelpers_asUint64(t *testing.T) {
	for _, v := range integerFactory(3) {
		assert.Equal(t, uint64(3), valueHelp.asUint64(v))
	}

	// Non numeric
	if assert.NotPanics(t, func() {
		valueHelp.asUint64("123")
	}) {
		assert.Equal(t, valueHelp.asUint64("123"), (uint64)(0))
	}
}

// Test cases in private method asFloat64()
func TestHelpers_asFloat64(t *testing.T) {
	for _, v := range integerFactory(3) {
		assert.Equal(t, float64(3), valueHelp.asFloat64(v))
	}

	// Non numeric
	if assert.NotPanics(t, func() {
		valueHelp.asFloat64("123")
	}) {
		assert.Equal(t, valueHelp.asFloat64("123"), (float64)(0))
	}
}
