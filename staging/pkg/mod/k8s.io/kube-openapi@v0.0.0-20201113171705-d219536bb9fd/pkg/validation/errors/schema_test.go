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
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestSchemaErrors(t *testing.T) {
	err := InvalidType("confirmed", "query", "boolean", nil)
	assert.Error(t, err)
	assert.EqualValues(t, InvalidTypeCode, err.Code())
	assert.Equal(t, "confirmed in query must be of type boolean", err.Error())

	err = InvalidType("confirmed", "", "boolean", nil)
	assert.Error(t, err)
	assert.EqualValues(t, InvalidTypeCode, err.Code())
	assert.Equal(t, "confirmed must be of type boolean", err.Error())

	err = InvalidType("confirmed", "query", "boolean", "hello")
	assert.Error(t, err)
	assert.EqualValues(t, InvalidTypeCode, err.Code())
	assert.Equal(t, "confirmed in query must be of type boolean: \"hello\"", err.Error())

	err = InvalidType("confirmed", "query", "boolean", errors.New("hello"))
	assert.Error(t, err)
	assert.EqualValues(t, InvalidTypeCode, err.Code())
	assert.Equal(t, "confirmed in query must be of type boolean, because: hello", err.Error())

	err = InvalidType("confirmed", "", "boolean", "hello")
	assert.Error(t, err)
	assert.EqualValues(t, InvalidTypeCode, err.Code())
	assert.Equal(t, "confirmed must be of type boolean: \"hello\"", err.Error())

	err = InvalidType("confirmed", "", "boolean", errors.New("hello"))
	assert.Error(t, err)
	assert.EqualValues(t, InvalidTypeCode, err.Code())
	assert.Equal(t, "confirmed must be of type boolean, because: hello", err.Error())

	err = DuplicateItems("uniques", "query")
	assert.Error(t, err)
	assert.EqualValues(t, UniqueFailCode, err.Code())
	assert.Equal(t, "uniques in query shouldn't contain duplicates", err.Error())

	err = DuplicateItems("uniques", "")
	assert.Error(t, err)
	assert.EqualValues(t, UniqueFailCode, err.Code())
	assert.Equal(t, "uniques shouldn't contain duplicates", err.Error())

	err = TooManyItems("something", "query", 5, 6)
	assert.Error(t, err)
	assert.EqualValues(t, MaxItemsFailCode, err.Code())
	assert.Equal(t, "something in query should have at most 5 items", err.Error())
	assert.Equal(t, 6, err.Value)

	err = TooManyItems("something", "", 5, 6)
	assert.Error(t, err)
	assert.EqualValues(t, MaxItemsFailCode, err.Code())
	assert.Equal(t, "something should have at most 5 items", err.Error())
	assert.Equal(t, 6, err.Value)

	err = TooFewItems("something", "", 5, 4)
	assert.Error(t, err)
	assert.EqualValues(t, MinItemsFailCode, err.Code())
	assert.Equal(t, "something should have at least 5 items", err.Error())
	assert.Equal(t, 4, err.Value)

	err = ExceedsMaximumInt("something", "query", 5, false, 6)
	assert.Error(t, err)
	assert.EqualValues(t, MaxFailCode, err.Code())
	assert.Equal(t, "something in query should be less than or equal to 5", err.Error())
	assert.Equal(t, 6, err.Value)

	err = ExceedsMaximumInt("something", "", 5, false, 6)
	assert.Error(t, err)
	assert.EqualValues(t, MaxFailCode, err.Code())
	assert.Equal(t, "something should be less than or equal to 5", err.Error())
	assert.Equal(t, 6, err.Value)

	err = ExceedsMaximumInt("something", "query", 5, true, 6)
	assert.Error(t, err)
	assert.EqualValues(t, MaxFailCode, err.Code())
	assert.Equal(t, "something in query should be less than 5", err.Error())
	assert.Equal(t, 6, err.Value)

	err = ExceedsMaximumInt("something", "", 5, true, 6)
	assert.Error(t, err)
	assert.EqualValues(t, MaxFailCode, err.Code())
	assert.Equal(t, "something should be less than 5", err.Error())
	assert.Equal(t, 6, err.Value)

	err = ExceedsMaximumUint("something", "query", 5, false, 6)
	assert.Error(t, err)
	assert.EqualValues(t, MaxFailCode, err.Code())
	assert.Equal(t, "something in query should be less than or equal to 5", err.Error())
	assert.Equal(t, 6, err.Value)

	err = ExceedsMaximumUint("something", "", 5, false, 6)
	assert.Error(t, err)
	assert.EqualValues(t, MaxFailCode, err.Code())
	assert.Equal(t, "something should be less than or equal to 5", err.Error())
	assert.Equal(t, 6, err.Value)

	err = ExceedsMaximumUint("something", "query", 5, true, 6)
	assert.Error(t, err)
	assert.EqualValues(t, MaxFailCode, err.Code())
	assert.Equal(t, "something in query should be less than 5", err.Error())
	assert.Equal(t, 6, err.Value)

	err = ExceedsMaximumUint("something", "", 5, true, 6)
	assert.Error(t, err)
	assert.EqualValues(t, MaxFailCode, err.Code())
	assert.Equal(t, "something should be less than 5", err.Error())
	assert.Equal(t, 6, err.Value)

	err = ExceedsMaximum("something", "query", 5, false, 6)
	assert.Error(t, err)
	assert.EqualValues(t, MaxFailCode, err.Code())
	assert.Equal(t, "something in query should be less than or equal to 5", err.Error())
	assert.Equal(t, 6, err.Value)

	err = ExceedsMaximum("something", "", 5, false, 6)
	assert.Error(t, err)
	assert.EqualValues(t, MaxFailCode, err.Code())
	assert.Equal(t, "something should be less than or equal to 5", err.Error())
	assert.Equal(t, 6, err.Value)

	err = ExceedsMaximum("something", "query", 5, true, 6)
	assert.Error(t, err)
	assert.EqualValues(t, MaxFailCode, err.Code())
	assert.Equal(t, "something in query should be less than 5", err.Error())
	assert.Equal(t, 6, err.Value)

	err = ExceedsMaximum("something", "", 5, true, 6)
	assert.Error(t, err)
	assert.EqualValues(t, MaxFailCode, err.Code())
	assert.Equal(t, "something should be less than 5", err.Error())
	assert.Equal(t, 6, err.Value)

	err = ExceedsMinimumInt("something", "query", 5, false, 4)
	assert.Error(t, err)
	assert.EqualValues(t, MinFailCode, err.Code())
	assert.Equal(t, "something in query should be greater than or equal to 5", err.Error())
	assert.Equal(t, 4, err.Value)

	err = ExceedsMinimumInt("something", "", 5, false, 4)
	assert.Error(t, err)
	assert.EqualValues(t, MinFailCode, err.Code())
	assert.Equal(t, "something should be greater than or equal to 5", err.Error())
	assert.Equal(t, 4, err.Value)

	err = ExceedsMinimumInt("something", "query", 5, true, 4)
	assert.Error(t, err)
	assert.EqualValues(t, MinFailCode, err.Code())
	assert.Equal(t, "something in query should be greater than 5", err.Error())
	assert.Equal(t, 4, err.Value)

	err = ExceedsMinimumInt("something", "", 5, true, 4)
	assert.Error(t, err)
	assert.EqualValues(t, MinFailCode, err.Code())
	assert.Equal(t, "something should be greater than 5", err.Error())
	assert.Equal(t, 4, err.Value)

	err = ExceedsMinimumUint("something", "query", 5, false, 4)
	assert.Error(t, err)
	assert.EqualValues(t, MinFailCode, err.Code())
	assert.Equal(t, "something in query should be greater than or equal to 5", err.Error())
	assert.Equal(t, 4, err.Value)

	err = ExceedsMinimumUint("something", "", 5, false, 4)
	assert.Error(t, err)
	assert.EqualValues(t, MinFailCode, err.Code())
	assert.Equal(t, "something should be greater than or equal to 5", err.Error())
	assert.Equal(t, 4, err.Value)

	err = ExceedsMinimumUint("something", "query", 5, true, 4)
	assert.Error(t, err)
	assert.EqualValues(t, MinFailCode, err.Code())
	assert.Equal(t, "something in query should be greater than 5", err.Error())
	assert.Equal(t, 4, err.Value)

	err = ExceedsMinimumUint("something", "", 5, true, 4)
	assert.Error(t, err)
	assert.EqualValues(t, MinFailCode, err.Code())
	assert.Equal(t, "something should be greater than 5", err.Error())
	assert.Equal(t, 4, err.Value)

	err = ExceedsMinimum("something", "query", 5, false, 4)
	assert.Error(t, err)
	assert.EqualValues(t, MinFailCode, err.Code())
	assert.Equal(t, "something in query should be greater than or equal to 5", err.Error())
	assert.Equal(t, 4, err.Value)

	err = ExceedsMinimum("something", "", 5, false, 4)
	assert.Error(t, err)
	assert.EqualValues(t, MinFailCode, err.Code())
	assert.Equal(t, "something should be greater than or equal to 5", err.Error())
	assert.Equal(t, 4, err.Value)

	err = ExceedsMinimum("something", "query", 5, true, 4)
	assert.Error(t, err)
	assert.EqualValues(t, MinFailCode, err.Code())
	assert.Equal(t, "something in query should be greater than 5", err.Error())
	assert.Equal(t, 4, err.Value)

	err = ExceedsMinimum("something", "", 5, true, 4)
	assert.Error(t, err)
	assert.EqualValues(t, MinFailCode, err.Code())
	assert.Equal(t, "something should be greater than 5", err.Error())
	assert.Equal(t, 4, err.Value)

	err = NotMultipleOf("something", "query", 5, 1)
	assert.Error(t, err)
	assert.EqualValues(t, MultipleOfFailCode, err.Code())
	assert.Equal(t, "something in query should be a multiple of 5", err.Error())
	assert.Equal(t, 1, err.Value)

	err = NotMultipleOf("something", "query", float64(5), float64(1))
	assert.Error(t, err)
	assert.EqualValues(t, MultipleOfFailCode, err.Code())
	assert.Equal(t, "something in query should be a multiple of 5", err.Error())
	assert.Equal(t, float64(1), err.Value)

	err = NotMultipleOf("something", "query", uint64(5), uint64(1))
	assert.Error(t, err)
	assert.EqualValues(t, MultipleOfFailCode, err.Code())
	assert.Equal(t, "something in query should be a multiple of 5", err.Error())
	assert.Equal(t, uint64(1), err.Value)

	err = NotMultipleOf("something", "", 5, 1)
	assert.Error(t, err)
	assert.EqualValues(t, MultipleOfFailCode, err.Code())
	assert.Equal(t, "something should be a multiple of 5", err.Error())
	assert.Equal(t, 1, err.Value)

	err = EnumFail("something", "query", "yada", []interface{}{"hello", "world"})
	assert.Error(t, err)
	assert.EqualValues(t, EnumFailCode, err.Code())
	assert.Equal(t, "something in query should be one of [hello world]", err.Error())
	assert.Equal(t, "yada", err.Value)

	err = EnumFail("something", "", "yada", []interface{}{"hello", "world"})
	assert.Error(t, err)
	assert.EqualValues(t, EnumFailCode, err.Code())
	assert.Equal(t, "something should be one of [hello world]", err.Error())
	assert.Equal(t, "yada", err.Value)

	err = Required("something", "query")
	assert.Error(t, err)
	assert.EqualValues(t, RequiredFailCode, err.Code())
	assert.Equal(t, "something in query is required", err.Error())

	err = Required("something", "")
	assert.Error(t, err)
	assert.EqualValues(t, RequiredFailCode, err.Code())
	assert.Equal(t, "something is required", err.Error())

	err = TooLong("something", "query", 5, "abcdef")
	assert.Error(t, err)
	assert.EqualValues(t, TooLongFailCode, err.Code())
	assert.Equal(t, "something in query should be at most 5 chars long", err.Error())
	assert.Equal(t, "abcdef", err.Value)

	err = TooLong("something", "", 5, "abcdef")
	assert.Error(t, err)
	assert.EqualValues(t, TooLongFailCode, err.Code())
	assert.Equal(t, "something should be at most 5 chars long", err.Error())
	assert.Equal(t, "abcdef", err.Value)

	err = TooShort("something", "query", 5, "a")
	assert.Error(t, err)
	assert.EqualValues(t, TooShortFailCode, err.Code())
	assert.Equal(t, "something in query should be at least 5 chars long", err.Error())
	assert.Equal(t, "a", err.Value)

	err = TooShort("something", "", 5, "a")
	assert.Error(t, err)
	assert.EqualValues(t, TooShortFailCode, err.Code())
	assert.Equal(t, "something should be at least 5 chars long", err.Error())
	assert.Equal(t, "a", err.Value)

	err = FailedPattern("something", "query", "\\d+", "a")
	assert.Error(t, err)
	assert.EqualValues(t, PatternFailCode, err.Code())
	assert.Equal(t, "something in query should match '\\d+'", err.Error())
	assert.Equal(t, "a", err.Value)

	err = FailedPattern("something", "", "\\d+", "a")
	assert.Error(t, err)
	assert.EqualValues(t, PatternFailCode, err.Code())
	assert.Equal(t, "something should match '\\d+'", err.Error())
	assert.Equal(t, "a", err.Value)

	err = InvalidTypeName("something")
	assert.Error(t, err)
	assert.EqualValues(t, InvalidTypeCode, err.Code())
	assert.Equal(t, "something is an invalid type name", err.Error())

	err = AdditionalItemsNotAllowed("something", "query")
	assert.Error(t, err)
	assert.EqualValues(t, NoAdditionalItemsCode, err.Code())
	assert.Equal(t, "something in query can't have additional items", err.Error())

	err = AdditionalItemsNotAllowed("something", "")
	assert.Error(t, err)
	assert.EqualValues(t, NoAdditionalItemsCode, err.Code())
	assert.Equal(t, "something can't have additional items", err.Error())

	err = InvalidCollectionFormat("something", "query", "yada")
	assert.Error(t, err)
	assert.EqualValues(t, InvalidTypeCode, err.Code())
	assert.Equal(t, "the collection format \"yada\" is not supported for the query param \"something\"", err.Error())

	err2 := CompositeValidationError()
	assert.Error(t, err2)
	assert.EqualValues(t, CompositeErrorCode, err2.Code())
	assert.Equal(t, "validation failure list", err2.Error())

	err2 = CompositeValidationError(fmt.Errorf("First error"), fmt.Errorf("Second error"))
	assert.Error(t, err2)
	assert.EqualValues(t, CompositeErrorCode, err2.Code())
	assert.Equal(t, "validation failure list:\nFirst error\nSecond error", err2.Error())

	//func MultipleOfMustBePositive(name, in string, factor interface{}) *Validation {
	err = MultipleOfMustBePositive("path", "body", float64(-10))
	assert.Error(t, err)
	assert.EqualValues(t, MultipleOfMustBePositiveCode, err.Code())
	assert.Equal(t, `factor MultipleOf declared for path must be positive: -10`, err.Error())
	assert.Equal(t, float64(-10), err.Value)

	err = MultipleOfMustBePositive("path", "body", int64(-10))
	assert.Error(t, err)
	assert.EqualValues(t, MultipleOfMustBePositiveCode, err.Code())
	assert.Equal(t, `factor MultipleOf declared for path must be positive: -10`, err.Error())
	assert.Equal(t, int64(-10), err.Value)

	// func PropertyNotAllowed(name, in, key string) *Validation {
	err = PropertyNotAllowed("path", "body", "key")
	assert.Error(t, err)
	assert.EqualValues(t, UnallowedPropertyCode, err.Code())
	//unallowedProperty         = "%s.%s in %s is a forbidden property"
	assert.Equal(t, "path.key in body is a forbidden property", err.Error())

	err = PropertyNotAllowed("path", "", "key")
	assert.Error(t, err)
	assert.EqualValues(t, UnallowedPropertyCode, err.Code())
	//unallowedPropertyNoIn     = "%s.%s is a forbidden property"
	assert.Equal(t, "path.key is a forbidden property", err.Error())

	//func TooManyProperties(name, in string, n int64) *Validation {
	err = TooManyProperties("path", "body", 10)
	assert.Error(t, err)
	assert.EqualValues(t, TooManyPropertiesCode, err.Code())
	//tooManyProperties         = "%s in %s should have at most %d properties"
	assert.Equal(t, "path in body should have at most 10 properties", err.Error())

	err = TooManyProperties("path", "", 10)
	assert.Error(t, err)
	assert.EqualValues(t, TooManyPropertiesCode, err.Code())
	//tooManyPropertiesNoIn     = "%s should have at most %d properties"
	assert.Equal(t, "path should have at most 10 properties", err.Error())

	err = TooFewProperties("path", "body", 10)
	// func TooFewProperties(name, in string, n int64) *Validation {
	assert.Error(t, err)
	assert.EqualValues(t, TooFewPropertiesCode, err.Code())
	//tooFewProperties          = "%s in %s should have at least %d properties"
	assert.Equal(t, "path in body should have at least 10 properties", err.Error())

	err = TooFewProperties("path", "", 10)
	// func TooFewProperties(name, in string, n int64) *Validation {
	assert.Error(t, err)
	assert.EqualValues(t, TooFewPropertiesCode, err.Code())
	//tooFewPropertiesNoIn      = "%s should have at least %d properties"
	assert.Equal(t, "path should have at least 10 properties", err.Error())

	//func FailedAllPatternProperties(name, in, key string) *Validation {
	err = FailedAllPatternProperties("path", "body", "key")
	assert.Error(t, err)
	assert.EqualValues(t, FailedAllPatternPropsCode, err.Code())
	//failedAllPatternProps     = "%s.%s in %s failed all pattern properties"
	assert.Equal(t, "path.key in body failed all pattern properties", err.Error())

	err = FailedAllPatternProperties("path", "", "key")
	assert.Error(t, err)
	assert.EqualValues(t, FailedAllPatternPropsCode, err.Code())
	//failedAllPatternPropsNoIn = "%s.%s failed all pattern properties"
	assert.Equal(t, "path.key failed all pattern properties", err.Error())
}
