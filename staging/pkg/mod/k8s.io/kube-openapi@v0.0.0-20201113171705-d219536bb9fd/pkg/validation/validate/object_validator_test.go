// Copyright 2017 go-swagger maintainers
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

func itemsFixture() map[string]interface{} {
	return map[string]interface{}{
		"type":  "array",
		"items": "dummy",
	}
}

func expectAllValid(t *testing.T, ov EntityValidator, dataValid, dataInvalid map[string]interface{}) {
	res := ov.Validate(dataValid)
	assert.Equal(t, 0, len(res.Errors))

	res = ov.Validate(dataInvalid)
	assert.Equal(t, 0, len(res.Errors))
}

func expectOnlyInvalid(t *testing.T, ov EntityValidator, dataValid, dataInvalid map[string]interface{}) {
	res := ov.Validate(dataValid)
	assert.Equal(t, 0, len(res.Errors))

	res = ov.Validate(dataInvalid)
	assert.NotEqual(t, 0, len(res.Errors))
}

func TestItemsMustBeTypeArray(t *testing.T) {
	ov := new(objectValidator)
	dataValid := itemsFixture()
	dataInvalid := map[string]interface{}{
		"type":  "object",
		"items": "dummy",
	}
	expectAllValid(t, ov, dataValid, dataInvalid)
}

func TestItemsMustHaveType(t *testing.T) {
	ov := new(objectValidator)
	dataValid := itemsFixture()
	dataInvalid := map[string]interface{}{
		"items": "dummy",
	}
	expectAllValid(t, ov, dataValid, dataInvalid)
}

func TestTypeArrayMustHaveItems(t *testing.T) {
	ov := new(objectValidator)
	dataValid := itemsFixture()
	dataInvalid := map[string]interface{}{
		"type": "array",
		"key":  "dummy",
	}
	expectAllValid(t, ov, dataValid, dataInvalid)
}

// Test edge cases in object_validator which are difficult
// to simulate with specs
// (this one is a trivial, just to check all methods are filled)
func TestObjectValidator_EdgeCases(t *testing.T) {
	s := objectValidator{}
	s.SetPath("path")
	assert.Equal(t, "path", s.Path)
}
