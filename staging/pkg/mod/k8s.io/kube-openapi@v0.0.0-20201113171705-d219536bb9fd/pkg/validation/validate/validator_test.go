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
	"reflect"
	"testing"

	"github.com/stretchr/testify/assert"
	"k8s.io/kube-openapi/pkg/validation/spec"
)

func TestNumberValidator_EdgeCases(t *testing.T) {
	// Apply
	var min = float64(math.MinInt32 - 1)
	var max = float64(math.MaxInt32 + 1)

	v := numberValidator{
		Path: "path",
		In:   "in",
		//Default:
		//MultipleOf:
		Maximum:          &max, // *float64
		ExclusiveMaximum: false,
		Minimum:          &min, // *float64
		ExclusiveMinimum: false,
		// Allows for more accurate behavior regarding integers
		Type:   "integer",
		Format: "int32",
	}

	// numberValidator applies to: Parameter,Schema,Items,Header

	sources := []interface{}{
		new(spec.Schema),
	}

	testNumberApply(t, &v, sources)

	assert.False(t, v.Applies(float64(32), reflect.Float64))

	// Now for different scenarios on Minimum, Maximum
	// - The Maximum value does not respect the Type|Format specification
	// - Value is checked as float64 with Maximum as float64 and fails
	res := v.Validate(int64(math.MaxInt32 + 2))
	assert.True(t, res.HasErrors())
	// - The Minimum value does not respect the Type|Format specification
	// - Value is checked as float64 with Maximum as float64 and fails
	res = v.Validate(int64(math.MinInt32 - 2))
	assert.True(t, res.HasErrors())
}

func testNumberApply(t *testing.T, v *numberValidator, sources []interface{}) {
	for _, source := range sources {
		// numberValidator does not applies to:
		assert.False(t, v.Applies(source, reflect.String))
		assert.False(t, v.Applies(source, reflect.Struct))
		// numberValidator applies to:
		assert.True(t, v.Applies(source, reflect.Int))
		assert.True(t, v.Applies(source, reflect.Int8))
		assert.True(t, v.Applies(source, reflect.Uint16))
		assert.True(t, v.Applies(source, reflect.Uint32))
		assert.True(t, v.Applies(source, reflect.Uint64))
		assert.True(t, v.Applies(source, reflect.Uint))
		assert.True(t, v.Applies(source, reflect.Uint8))
		assert.True(t, v.Applies(source, reflect.Uint16))
		assert.True(t, v.Applies(source, reflect.Uint32))
		assert.True(t, v.Applies(source, reflect.Uint64))
		assert.True(t, v.Applies(source, reflect.Float32))
		assert.True(t, v.Applies(source, reflect.Float64))
	}
}

func TestStringValidator_EdgeCases(t *testing.T) {
	// Apply

	v := stringValidator{}

	// stringValidator applies to: Parameter,Schema,Items,Header

	sources := []interface{}{
		new(spec.Schema),
	}

	testStringApply(t, &v, sources)

	assert.False(t, v.Applies("A string", reflect.String))

}

func testStringApply(t *testing.T, v *stringValidator, sources []interface{}) {
	for _, source := range sources {
		// numberValidator does not applies to:
		assert.False(t, v.Applies(source, reflect.Struct))
		assert.False(t, v.Applies(source, reflect.Int))
		// numberValidator applies to:
		assert.True(t, v.Applies(source, reflect.String))
	}
}

func TestBasicCommonValidator_EdgeCases(t *testing.T) {
	// Apply

	v := basicCommonValidator{}

	// basicCommonValidator applies to: Parameter,Schema,Header

	sources := []interface{}{
		new(spec.Schema),
	}

	testCommonApply(t, &v, sources)

	assert.False(t, v.Applies("A string", reflect.String))

}

func testCommonApply(t *testing.T, v *basicCommonValidator, sources []interface{}) {
	for _, source := range sources {
		assert.True(t, v.Applies(source, reflect.String))
	}
}
