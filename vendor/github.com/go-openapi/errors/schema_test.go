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

package errors

import (
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestSchemaErrors(t *testing.T) {
	err := InvalidType("confirmed", "query", "boolean", nil)
	assert.Error(t, err)
	assert.EqualValues(t, 422, err.Code())
	assert.Equal(t, "confirmed in query must be of type boolean", err.Error())

	err = InvalidType("confirmed", "", "boolean", nil)
	assert.Error(t, err)
	assert.EqualValues(t, 422, err.Code())
	assert.Equal(t, "confirmed must be of type boolean", err.Error())

	err = InvalidType("confirmed", "query", "boolean", "hello")
	assert.Error(t, err)
	assert.EqualValues(t, 422, err.Code())
	assert.Equal(t, "confirmed in query must be of type boolean: \"hello\"", err.Error())

	err = InvalidType("confirmed", "query", "boolean", errors.New("hello"))
	assert.Error(t, err)
	assert.EqualValues(t, 422, err.Code())
	assert.Equal(t, "confirmed in query must be of type boolean, because: hello", err.Error())

	err = InvalidType("confirmed", "", "boolean", "hello")
	assert.Error(t, err)
	assert.EqualValues(t, 422, err.Code())
	assert.Equal(t, "confirmed must be of type boolean: \"hello\"", err.Error())

	err = InvalidType("confirmed", "", "boolean", errors.New("hello"))
	assert.Error(t, err)
	assert.EqualValues(t, 422, err.Code())
	assert.Equal(t, "confirmed must be of type boolean, because: hello", err.Error())

	err = DuplicateItems("uniques", "query")
	assert.Error(t, err)
	assert.EqualValues(t, 422, err.Code())
	assert.Equal(t, "uniques in query shouldn't contain duplicates", err.Error())

	err = DuplicateItems("uniques", "")
	assert.Error(t, err)
	assert.EqualValues(t, 422, err.Code())
	assert.Equal(t, "uniques shouldn't contain duplicates", err.Error())

	err = TooManyItems("something", "query", 5)
	assert.Error(t, err)
	assert.EqualValues(t, 422, err.Code())
	assert.Equal(t, "something in query should have at most 5 items", err.Error())

	err = TooManyItems("something", "", 5)
	assert.Error(t, err)
	assert.EqualValues(t, 422, err.Code())
	assert.Equal(t, "something should have at most 5 items", err.Error())

	err = TooFewItems("something", "", 5)
	assert.Error(t, err)
	assert.EqualValues(t, 422, err.Code())
	assert.Equal(t, "something should have at least 5 items", err.Error())

	err = ExceedsMaximumInt("something", "query", 5, false)
	assert.Error(t, err)
	assert.EqualValues(t, 422, err.Code())
	assert.Equal(t, "something in query should be less than or equal to 5", err.Error())

	err = ExceedsMaximumInt("something", "", 5, false)
	assert.Error(t, err)
	assert.EqualValues(t, 422, err.Code())
	assert.Equal(t, "something should be less than or equal to 5", err.Error())

	err = ExceedsMaximumInt("something", "query", 5, true)
	assert.Error(t, err)
	assert.EqualValues(t, 422, err.Code())
	assert.Equal(t, "something in query should be less than 5", err.Error())

	err = ExceedsMaximumInt("something", "", 5, true)
	assert.Error(t, err)
	assert.EqualValues(t, 422, err.Code())
	assert.Equal(t, "something should be less than 5", err.Error())

	err = ExceedsMaximumUint("something", "query", 5, false)
	assert.Error(t, err)
	assert.EqualValues(t, 422, err.Code())
	assert.Equal(t, "something in query should be less than or equal to 5", err.Error())

	err = ExceedsMaximumUint("something", "", 5, false)
	assert.Error(t, err)
	assert.EqualValues(t, 422, err.Code())
	assert.Equal(t, "something should be less than or equal to 5", err.Error())

	err = ExceedsMaximumUint("something", "query", 5, true)
	assert.Error(t, err)
	assert.EqualValues(t, 422, err.Code())
	assert.Equal(t, "something in query should be less than 5", err.Error())

	err = ExceedsMaximumUint("something", "", 5, true)
	assert.Error(t, err)
	assert.EqualValues(t, 422, err.Code())
	assert.Equal(t, "something should be less than 5", err.Error())

	err = ExceedsMaximum("something", "query", 5, false)
	assert.Error(t, err)
	assert.EqualValues(t, 422, err.Code())
	assert.Equal(t, "something in query should be less than or equal to 5", err.Error())

	err = ExceedsMaximum("something", "", 5, false)
	assert.Error(t, err)
	assert.EqualValues(t, 422, err.Code())
	assert.Equal(t, "something should be less than or equal to 5", err.Error())

	err = ExceedsMaximum("something", "query", 5, true)
	assert.Error(t, err)
	assert.EqualValues(t, 422, err.Code())
	assert.Equal(t, "something in query should be less than 5", err.Error())

	err = ExceedsMaximum("something", "", 5, true)
	assert.Error(t, err)
	assert.EqualValues(t, 422, err.Code())
	assert.Equal(t, "something should be less than 5", err.Error())

	err = ExceedsMinimumInt("something", "query", 5, false)
	assert.Error(t, err)
	assert.EqualValues(t, 422, err.Code())
	assert.Equal(t, "something in query should be greater than or equal to 5", err.Error())

	err = ExceedsMinimumInt("something", "", 5, false)
	assert.Error(t, err)
	assert.EqualValues(t, 422, err.Code())
	assert.Equal(t, "something should be greater than or equal to 5", err.Error())

	err = ExceedsMinimumInt("something", "query", 5, true)
	assert.Error(t, err)
	assert.EqualValues(t, 422, err.Code())
	assert.Equal(t, "something in query should be greater than 5", err.Error())

	err = ExceedsMinimumInt("something", "", 5, true)
	assert.Error(t, err)
	assert.EqualValues(t, 422, err.Code())
	assert.Equal(t, "something should be greater than 5", err.Error())

	err = ExceedsMinimumUint("something", "query", 5, false)
	assert.Error(t, err)
	assert.EqualValues(t, 422, err.Code())
	assert.Equal(t, "something in query should be greater than or equal to 5", err.Error())

	err = ExceedsMinimumUint("something", "", 5, false)
	assert.Error(t, err)
	assert.EqualValues(t, 422, err.Code())
	assert.Equal(t, "something should be greater than or equal to 5", err.Error())

	err = ExceedsMinimumUint("something", "query", 5, true)
	assert.Error(t, err)
	assert.EqualValues(t, 422, err.Code())
	assert.Equal(t, "something in query should be greater than 5", err.Error())

	err = ExceedsMinimumUint("something", "", 5, true)
	assert.Error(t, err)
	assert.EqualValues(t, 422, err.Code())
	assert.Equal(t, "something should be greater than 5", err.Error())

	err = ExceedsMinimum("something", "query", 5, false)
	assert.Error(t, err)
	assert.EqualValues(t, 422, err.Code())
	assert.Equal(t, "something in query should be greater than or equal to 5", err.Error())

	err = ExceedsMinimum("something", "", 5, false)
	assert.Error(t, err)
	assert.EqualValues(t, 422, err.Code())
	assert.Equal(t, "something should be greater than or equal to 5", err.Error())

	err = ExceedsMinimum("something", "query", 5, true)
	assert.Error(t, err)
	assert.EqualValues(t, 422, err.Code())
	assert.Equal(t, "something in query should be greater than 5", err.Error())

	err = ExceedsMinimum("something", "", 5, true)
	assert.Error(t, err)
	assert.EqualValues(t, 422, err.Code())
	assert.Equal(t, "something should be greater than 5", err.Error())

	err = NotMultipleOf("something", "query", 5)
	assert.Error(t, err)
	assert.EqualValues(t, 422, err.Code())
	assert.Equal(t, "something in query should be a multiple of 5", err.Error())

	err = NotMultipleOf("something", "", 5)
	assert.Error(t, err)
	assert.EqualValues(t, 422, err.Code())
	assert.Equal(t, "something should be a multiple of 5", err.Error())

	err = EnumFail("something", "query", "yada", []interface{}{"hello", "world"})
	assert.Error(t, err)
	assert.EqualValues(t, 422, err.Code())
	assert.Equal(t, "something in query should be one of [hello world]", err.Error())

	err = EnumFail("something", "", "yada", []interface{}{"hello", "world"})
	assert.Error(t, err)
	assert.EqualValues(t, 422, err.Code())
	assert.Equal(t, "something should be one of [hello world]", err.Error())

	err = Required("something", "query")
	assert.Error(t, err)
	assert.EqualValues(t, 422, err.Code())
	assert.Equal(t, "something in query is required", err.Error())

	err = Required("something", "")
	assert.Error(t, err)
	assert.EqualValues(t, 422, err.Code())
	assert.Equal(t, "something is required", err.Error())

	err = TooLong("something", "query", 5)
	assert.Error(t, err)
	assert.EqualValues(t, 422, err.Code())
	assert.Equal(t, "something in query should be at most 5 chars long", err.Error())

	err = TooLong("something", "", 5)
	assert.Error(t, err)
	assert.EqualValues(t, 422, err.Code())
	assert.Equal(t, "something should be at most 5 chars long", err.Error())

	err = TooShort("something", "query", 5)
	assert.Error(t, err)
	assert.EqualValues(t, 422, err.Code())
	assert.Equal(t, "something in query should be at least 5 chars long", err.Error())

	err = TooShort("something", "", 5)
	assert.Error(t, err)
	assert.EqualValues(t, 422, err.Code())
	assert.Equal(t, "something should be at least 5 chars long", err.Error())

	err = FailedPattern("something", "query", "\\d+")
	assert.Error(t, err)
	assert.EqualValues(t, 422, err.Code())
	assert.Equal(t, "something in query should match '\\d+'", err.Error())

	err = FailedPattern("something", "", "\\d+")
	assert.Error(t, err)
	assert.EqualValues(t, 422, err.Code())
	assert.Equal(t, "something should match '\\d+'", err.Error())

	err = InvalidTypeName("something")
	assert.Error(t, err)
	assert.EqualValues(t, 422, err.Code())
	assert.Equal(t, "something is an invalid type name", err.Error())

	err = AdditionalItemsNotAllowed("something", "query")
	assert.Error(t, err)
	assert.EqualValues(t, 422, err.Code())
	assert.Equal(t, "something in query can't have additional items", err.Error())

	err = AdditionalItemsNotAllowed("something", "")
	assert.Error(t, err)
	assert.EqualValues(t, 422, err.Code())
	assert.Equal(t, "something can't have additional items", err.Error())

	err = InvalidCollectionFormat("something", "query", "yada")
	assert.Error(t, err)
	assert.EqualValues(t, 422, err.Code())
	assert.Equal(t, "the collection format \"yada\" is not supported for the query param \"something\"", err.Error())

	err2 := CompositeValidationError()
	assert.Error(t, err2)
	assert.EqualValues(t, 422, err2.Code())
	assert.Equal(t, "validation failure list", err2.Error())
}
