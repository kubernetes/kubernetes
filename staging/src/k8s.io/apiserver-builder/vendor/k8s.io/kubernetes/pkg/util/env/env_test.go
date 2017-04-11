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

func TestGetEnvAsStringOrFallback(t *testing.T) {
	const expected = "foo"

	assert := assert.New(t)

	key := "FLOCKER_SET_VAR"
	os.Setenv(key, expected)
	assert.Equal(expected, GetEnvAsStringOrFallback(key, "~"+expected))

	key = "FLOCKER_UNSET_VAR"
	assert.Equal(expected, GetEnvAsStringOrFallback(key, expected))
}

func TestGetEnvAsIntOrFallback(t *testing.T) {
	const expected = 1

	assert := assert.New(t)

	key := "FLOCKER_SET_VAR"
	os.Setenv(key, strconv.Itoa(expected))
	returnVal, _ := GetEnvAsIntOrFallback(key, 1)
	assert.Equal(expected, returnVal)

	key = "FLOCKER_UNSET_VAR"
	returnVal, _ = GetEnvAsIntOrFallback(key, expected)
	assert.Equal(expected, returnVal)

	key = "FLOCKER_SET_VAR"
	os.Setenv(key, "not-an-int")
	returnVal, err := GetEnvAsIntOrFallback(key, 1)
	assert.Equal(expected, returnVal)
	assert.EqualError(err, "strconv.ParseInt: parsing \"not-an-int\": invalid syntax")
}

func TestGetEnvAsFloat64OrFallback(t *testing.T) {
	const expected = 1.0

	assert := assert.New(t)

	key := "FLOCKER_SET_VAR"
	os.Setenv(key, "1.0")
	returnVal, _ := GetEnvAsFloat64OrFallback(key, 2.0)
	assert.Equal(expected, returnVal)

	key = "FLOCKER_UNSET_VAR"
	returnVal, _ = GetEnvAsFloat64OrFallback(key, 1.0)
	assert.Equal(expected, returnVal)

	key = "FLOCKER_SET_VAR"
	os.Setenv(key, "not-a-float")
	returnVal, err := GetEnvAsFloat64OrFallback(key, 1.0)
	assert.Equal(expected, returnVal)
	assert.EqualError(err, "strconv.ParseFloat: parsing \"not-a-float\": invalid syntax")
}
