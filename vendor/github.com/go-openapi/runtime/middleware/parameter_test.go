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

package middleware

import (
	"math"
	"net/url"
	"reflect"
	"strconv"
	"testing"

	"github.com/go-openapi/errors"
	"github.com/go-openapi/runtime"
	"github.com/go-openapi/spec"
	"github.com/go-openapi/strfmt"
	"github.com/stretchr/testify/assert"
)

type email struct {
	Address string
}

type paramFactory func(string) *spec.Parameter

var paramFactories = []paramFactory{
	spec.QueryParam,
	spec.HeaderParam,
	spec.PathParam,
	spec.FormDataParam,
}

func np(param *spec.Parameter) *untypedParamBinder {
	return newUntypedParamBinder(*param, new(spec.Swagger), strfmt.Default)
}

var stringItems = new(spec.Items)

func init() {
	stringItems.Type = "string"
}

func testCollectionFormat(t *testing.T, param *spec.Parameter, valid bool) {
	binder := &untypedParamBinder{
		parameter: param,
	}
	_, _, _, err := binder.readValue(runtime.Values(nil), reflect.ValueOf(nil))
	if valid {
		assert.NoError(t, err)
	} else {
		assert.Error(t, err)
		assert.Equal(t, errors.InvalidCollectionFormat(param.Name, param.In, param.CollectionFormat), err)
	}
}

func requiredError(param *spec.Parameter) *errors.Validation {
	return errors.Required(param.Name, param.In)
}

func validateRequiredTest(t *testing.T, param *spec.Parameter, value reflect.Value) {

	binder := np(param)
	err := binder.bindValue([]string{}, true, value)
	assert.Error(t, err)
	assert.NotNil(t, param)
	assert.EqualError(t, requiredError(param), err.Error())
	err = binder.bindValue([]string{""}, true, value)
	if assert.Error(t, err) {
		assert.EqualError(t, requiredError(param), err.Error())
	}

	// should be impossible data, but let's go with it
	err = binder.bindValue([]string{"a"}, false, value)
	assert.Error(t, err)
	assert.EqualError(t, requiredError(param), err.Error())
	err = binder.bindValue([]string{""}, false, value)
	assert.Error(t, err)
	assert.EqualError(t, requiredError(param), err.Error())
}

func validateRequiredAllowEmptyTest(t *testing.T, param *spec.Parameter, value reflect.Value) {
	param.AllowEmptyValue = true
	binder := np(param)
	err := binder.bindValue([]string{}, true, value)
	assert.NoError(t, err)
	if assert.NotNil(t, param) {
		err = binder.bindValue([]string{""}, true, value)
		assert.NoError(t, err)
		err = binder.bindValue([]string{"1"}, false, value)
		assert.Error(t, err)
		assert.EqualError(t, requiredError(param), err.Error())
		err = binder.bindValue([]string{""}, false, value)
		assert.Error(t, err)
		assert.EqualError(t, requiredError(param), err.Error())
	}
}

func TestRequiredValidation(t *testing.T) {
	strParam := spec.QueryParam("name").Typed("string", "").AsRequired()
	validateRequiredTest(t, strParam, reflect.ValueOf(""))
	validateRequiredAllowEmptyTest(t, strParam, reflect.ValueOf(""))

	intParam := spec.QueryParam("id").Typed("integer", "int32").AsRequired()
	validateRequiredTest(t, intParam, reflect.ValueOf(int32(0)))
	validateRequiredAllowEmptyTest(t, intParam, reflect.ValueOf(int32(0)))
	longParam := spec.QueryParam("id").Typed("integer", "int64").AsRequired()
	validateRequiredTest(t, longParam, reflect.ValueOf(int64(0)))
	validateRequiredAllowEmptyTest(t, longParam, reflect.ValueOf(int64(0)))

	floatParam := spec.QueryParam("score").Typed("number", "float").AsRequired()
	validateRequiredTest(t, floatParam, reflect.ValueOf(float32(0)))
	validateRequiredAllowEmptyTest(t, floatParam, reflect.ValueOf(float32(0)))
	doubleParam := spec.QueryParam("score").Typed("number", "double").AsRequired()
	validateRequiredTest(t, doubleParam, reflect.ValueOf(float64(0)))
	validateRequiredAllowEmptyTest(t, doubleParam, reflect.ValueOf(float64(0)))

	dateTimeParam := spec.QueryParam("registered").Typed("string", "date-time").AsRequired()
	validateRequiredTest(t, dateTimeParam, reflect.ValueOf(strfmt.DateTime{}))
	// validateRequiredAllowEmptyTest(t, dateTimeParam, reflect.ValueOf(strfmt.DateTime{}))

	dateParam := spec.QueryParam("registered").Typed("string", "date").AsRequired()
	validateRequiredTest(t, dateParam, reflect.ValueOf(strfmt.Date{}))
	// validateRequiredAllowEmptyTest(t, dateParam, reflect.ValueOf(strfmt.DateTime{}))

	sliceParam := spec.QueryParam("tags").CollectionOf(stringItems, "").AsRequired()
	validateRequiredTest(t, sliceParam, reflect.MakeSlice(reflect.TypeOf([]string{}), 0, 0))
	validateRequiredAllowEmptyTest(t, sliceParam, reflect.MakeSlice(reflect.TypeOf([]string{}), 0, 0))
}

func TestInvalidCollectionFormat(t *testing.T) {
	validCf1 := spec.QueryParam("validFmt").CollectionOf(stringItems, "multi")
	validCf2 := spec.FormDataParam("validFmt2").CollectionOf(stringItems, "multi")
	invalidCf1 := spec.HeaderParam("invalidHdr").CollectionOf(stringItems, "multi")
	invalidCf2 := spec.PathParam("invalidPath").CollectionOf(stringItems, "multi")

	testCollectionFormat(t, validCf1, true)
	testCollectionFormat(t, validCf2, true)
	testCollectionFormat(t, invalidCf1, false)
	testCollectionFormat(t, invalidCf2, false)
}

func invalidTypeError(param *spec.Parameter, data interface{}) *errors.Validation {
	tpe := param.Type
	if param.Format != "" {
		tpe = param.Format
	}
	return errors.InvalidType(param.Name, param.In, tpe, data)
}

func TestTypeValidation(t *testing.T) {
	for _, newParam := range paramFactories {
		intParam := newParam("badInt").Typed("integer", "int32")
		value := reflect.ValueOf(int32(0))
		binder := np(intParam)
		err := binder.bindValue([]string{"yada"}, true, value)
		// fails for invalid string
		assert.Error(t, err)
		assert.Equal(t, invalidTypeError(intParam, "yada"), err)
		// fails for overflow
		val := int64(math.MaxInt32)
		str := strconv.FormatInt(val, 10) + "0"
		v := int32(0)
		value = reflect.ValueOf(&v).Elem()
		binder = np(intParam)
		err = binder.bindValue([]string{str}, true, value)
		assert.Error(t, err)
		assert.Equal(t, invalidTypeError(intParam, str), err)

		longParam := newParam("badLong").Typed("integer", "int64")
		value = reflect.ValueOf(int64(0))
		binder = np(longParam)
		err = binder.bindValue([]string{"yada"}, true, value)
		// fails for invalid string
		assert.Error(t, err)
		assert.Equal(t, invalidTypeError(longParam, "yada"), err)
		// fails for overflow
		str2 := strconv.FormatInt(math.MaxInt64, 10) + "0"
		v2 := int64(0)
		vv2 := reflect.ValueOf(&v2).Elem()
		binder = np(longParam)
		err = binder.bindValue([]string{str2}, true, vv2)
		assert.Error(t, err)
		assert.Equal(t, invalidTypeError(longParam, str2), err)

		floatParam := newParam("badFloat").Typed("number", "float")
		value = reflect.ValueOf(float64(0))
		binder = np(floatParam)
		err = binder.bindValue([]string{"yada"}, true, value)
		// fails for invalid string
		assert.Error(t, err)
		assert.Equal(t, invalidTypeError(floatParam, "yada"), err)
		// fails for overflow
		str3 := strconv.FormatFloat(math.MaxFloat64, 'f', 5, 64)
		v3 := reflect.TypeOf(float32(0))
		value = reflect.New(v3).Elem()
		binder = np(floatParam)
		err = binder.bindValue([]string{str3}, true, value)
		assert.Error(t, err)
		assert.Equal(t, invalidTypeError(floatParam, str3), err)

		doubleParam := newParam("badDouble").Typed("number", "double")
		value = reflect.ValueOf(float64(0))
		binder = np(doubleParam)
		err = binder.bindValue([]string{"yada"}, true, value)
		// fails for invalid string
		assert.Error(t, err)
		assert.Equal(t, invalidTypeError(doubleParam, "yada"), err)
		// fails for overflow
		str4 := "9" + strconv.FormatFloat(math.MaxFloat64, 'f', 5, 64)
		v4 := reflect.TypeOf(float64(0))
		value = reflect.New(v4).Elem()
		binder = np(doubleParam)
		err = binder.bindValue([]string{str4}, true, value)
		assert.Error(t, err)
		assert.Equal(t, invalidTypeError(doubleParam, str4), err)

		dateParam := newParam("badDate").Typed("string", "date")
		value = reflect.ValueOf(strfmt.Date{})
		binder = np(dateParam)
		err = binder.bindValue([]string{"yada"}, true, value)
		// fails for invalid string
		assert.Error(t, err)
		assert.Equal(t, invalidTypeError(dateParam, "yada"), err)

		dateTimeParam := newParam("badDateTime").Typed("string", "date-time")
		value = reflect.ValueOf(strfmt.DateTime{})
		binder = np(dateTimeParam)
		err = binder.bindValue([]string{"yada"}, true, value)
		// fails for invalid string
		assert.Error(t, err)
		assert.Equal(t, invalidTypeError(dateTimeParam, "yada"), err)

		byteParam := newParam("badByte").Typed("string", "byte")
		values := url.Values(map[string][]string{})
		values.Add("badByte", "yaüda")
		v5 := []byte{}
		value = reflect.ValueOf(&v5).Elem()
		binder = np(byteParam)
		err = binder.bindValue([]string{"yaüda"}, true, value)
		// fails for invalid string
		assert.Error(t, err)
		assert.Equal(t, invalidTypeError(byteParam, "yaüda"), err)
	}
}

func TestTypeDetectionInvalidItems(t *testing.T) {
	withoutItems := spec.QueryParam("without").CollectionOf(nil, "")
	binder := &untypedParamBinder{
		Name:      "without",
		parameter: withoutItems,
	}
	assert.Nil(t, binder.Type())

	items := new(spec.Items)
	items.Type = "array"
	withInvalidItems := spec.QueryParam("invalidItems").CollectionOf(items, "")
	binder = &untypedParamBinder{
		Name:      "invalidItems",
		parameter: withInvalidItems,
	}
	assert.Nil(t, binder.Type())

	noType := spec.QueryParam("invalidType")
	noType.Type = "invalid"
	binder = &untypedParamBinder{
		Name:      "invalidType",
		parameter: noType,
	}
	assert.Nil(t, binder.Type())
}

// type emailStrFmt struct {
// 	name      string
// 	tpe       reflect.Type
// 	validator FormatValidator
// }
//
// func (e *emailStrFmt) Name() string {
// 	return e.name
// }
//
// func (e *emailStrFmt) Type() reflect.Type {
// 	return e.tpe
// }
//
// func (e *emailStrFmt) Matches(str string) bool {
// 	return e.validator(str)
// }
//
// func TestTypeDetectionValid(t *testing.T) {
// 	// emlFmt := &emailStrFmt{
// 	// 	name: "email",
// 	// 	tpe:  reflect.TypeOf(email{}),
// 	// }
// 	// formats := []StringFormat{emlFmt}
//
// 	expected := map[string]reflect.Type{
// 		"name":         reflect.TypeOf(""),
// 		"id":           reflect.TypeOf(int64(0)),
// 		"age":          reflect.TypeOf(int32(0)),
// 		"score":        reflect.TypeOf(float32(0)),
// 		"factor":       reflect.TypeOf(float64(0)),
// 		"friend":       reflect.TypeOf(map[string]interface{}{}),
// 		"X-Request-Id": reflect.TypeOf(int64(0)),
// 		"tags":         reflect.TypeOf([]string{}),
// 		"confirmed":    reflect.TypeOf(true),
// 		"planned":      reflect.TypeOf(swagger.Date{}),
// 		"delivered":    reflect.TypeOf(swagger.DateTime{}),
// 		"email":        reflect.TypeOf(email{}),
// 		"picture":      reflect.TypeOf([]byte{}),
// 		"file":         reflect.TypeOf(&swagger.File{}).Elem(),
// 	}
//
// 	params := parametersForAllTypes("")
// 	emailParam := spec.QueryParam("email").Typed("string", "email")
// 	params["email"] = *emailParam
//
// 	fileParam := spec.FileParam("file")
// 	params["file"] = *fileParam
//
// 	for _, v := range params {
// 		binder := &paramBinder{
// 			formats:   formats,
// 			name:      v.Name,
// 			parameter: &v,
// 		}
// 		assert.Equal(t, expected[v.Name], binder.Type(), "name: %s", v.Name)
// 	}
// }
