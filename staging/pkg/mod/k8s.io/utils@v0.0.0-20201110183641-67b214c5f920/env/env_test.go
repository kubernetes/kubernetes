/*
Copyright 2015 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package env

import (
	"os"
	"strconv"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestGetString(t *testing.T) {
	const expected = "foo"

	assert := assert.New(t)

	key := "STRING_SET_VAR"
	os.Setenv(key, expected)
	assert.Equal(expected, GetString(key, "~"+expected))

	key = "STRING_UNSET_VAR"
	assert.Equal(expected, GetString(key, expected))
}

func TestGetInt(t *testing.T) {
	const expected = 1
	const defaultValue = 2

	assert := assert.New(t)

	key := "INT_SET_VAR"
	os.Setenv(key, strconv.Itoa(expected))
	returnVal, _ := GetInt(key, defaultValue)
	assert.Equal(expected, returnVal)

	key = "INT_UNSET_VAR"
	returnVal, _ = GetInt(key, defaultValue)
	assert.Equal(defaultValue, returnVal)

	key = "INT_SET_VAR"
	os.Setenv(key, "not-an-int")
	returnVal, err := GetInt(key, defaultValue)
	assert.Equal(defaultValue, returnVal)
	if err == nil {
		t.Error("expected error")
	}
}

func TestGetFloat64(t *testing.T) {
	const expected = 1.0
	const defaultValue = 2.0

	assert := assert.New(t)

	key := "FLOAT_SET_VAR"
	os.Setenv(key, "1.0")
	returnVal, _ := GetFloat64(key, defaultValue)
	assert.Equal(expected, returnVal)

	key = "FLOAT_UNSET_VAR"
	returnVal, _ = GetFloat64(key, defaultValue)
	assert.Equal(defaultValue, returnVal)

	key = "FLOAT_SET_VAR"
	os.Setenv(key, "not-a-float")
	returnVal, err := GetFloat64(key, defaultValue)
	assert.Equal(defaultValue, returnVal)
	assert.EqualError(err, "strconv.ParseFloat: parsing \"not-a-float\": invalid syntax")
}

func TestGetBool(t *testing.T) {
	const expected = true
	const defaultValue = false

	assert := assert.New(t)

	key := "BOOL_SET_VAR"
	os.Setenv(key, "true")
	returnVal, _ := GetBool(key, defaultValue)
	assert.Equal(expected, returnVal)

	key = "BOOL_UNSET_VAR"
	returnVal, _ = GetBool(key, defaultValue)
	assert.Equal(defaultValue, returnVal)

	key = "BOOL_SET_VAR"
	os.Setenv(key, "not-a-bool")
	returnVal, err := GetBool(key, defaultValue)
	assert.Equal(defaultValue, returnVal)
	assert.EqualError(err, "strconv.ParseBool: parsing \"not-a-bool\": invalid syntax")
}
