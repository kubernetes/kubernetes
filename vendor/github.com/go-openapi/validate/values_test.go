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

func TestValidateIntEnum(t *testing.T) {
	enumValues := []interface{}{1, 2, 3}

	err := Enum("test", "body", int64(5), enumValues)
	assert.Error(t, err)
	err = Enum("test", "body", int64(1), enumValues)
	assert.NoError(t, err)
}

func TestValidateEnum(t *testing.T) {
	enumValues := []string{"aa", "bb", "cc"}

	err := Enum("test", "body", "a", enumValues)
	assert.Error(t, err)
	err = Enum("test", "body", "bb", enumValues)
	assert.NoError(t, err)
}

func TestValidateUniqueItems(t *testing.T) {
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
		assert.NoError(t, err)
	}
}

func TestValidateMinLength(t *testing.T) {
	var minLength int64 = 5
	err := MinLength("test", "body", "aa", minLength)
	assert.Error(t, err)
	err = MinLength("test", "body", "aaaaa", minLength)
	assert.NoError(t, err)
}

func TestValidateMaxLength(t *testing.T) {
	var maxLength int64 = 5
	err := MaxLength("test", "body", "bbbbbb", maxLength)
	assert.Error(t, err)
	err = MaxLength("test", "body", "aa", maxLength)
	assert.NoError(t, err)
}

func TestValidateRequired(t *testing.T) {
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
		assert.NoError(t, err)
	}

}

func TestValidateRequiredNumber(t *testing.T) {
	err := RequiredNumber("test", "body", 0)
	assert.Error(t, err)
	err = RequiredNumber("test", "body", 1)
	assert.NoError(t, err)
}
