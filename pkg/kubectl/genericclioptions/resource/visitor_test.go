/*
Copyright 2016 The Kubernetes Authors.

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

package resource

import (
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestVisitorHttpGet(t *testing.T) {
	// Test retries on errors
	i := 0
	expectedErr := fmt.Errorf("Failed to get http")
	actualBytes, actualErr := readHttpWithRetries(func(url string) (int, string, io.ReadCloser, error) {
		assert.Equal(t, "hello", url)
		i++
		if i > 2 {
			return 0, "", nil, expectedErr
		}
		return 0, "", nil, fmt.Errorf("Unexpected error")
	}, 0, "hello", 3)
	assert.Equal(t, expectedErr, actualErr)
	assert.Nil(t, actualBytes)
	assert.Equal(t, 3, i)

	// Test that 500s are retried.
	i = 0
	actualBytes, actualErr = readHttpWithRetries(func(url string) (int, string, io.ReadCloser, error) {
		assert.Equal(t, "hello", url)
		i++
		return 501, "Status", nil, nil
	}, 0, "hello", 3)
	assert.Error(t, actualErr)
	assert.Nil(t, actualBytes)
	assert.Equal(t, 3, i)

	// Test that 300s are not retried
	i = 0
	actualBytes, actualErr = readHttpWithRetries(func(url string) (int, string, io.ReadCloser, error) {
		assert.Equal(t, "hello", url)
		i++
		return 300, "Status", nil, nil
	}, 0, "hello", 3)
	assert.Error(t, actualErr)
	assert.Nil(t, actualBytes)
	assert.Equal(t, 1, i)

	// Test attempt count is respected
	i = 0
	actualBytes, actualErr = readHttpWithRetries(func(url string) (int, string, io.ReadCloser, error) {
		assert.Equal(t, "hello", url)
		i++
		return 501, "Status", nil, nil
	}, 0, "hello", 1)
	assert.Error(t, actualErr)
	assert.Nil(t, actualBytes)
	assert.Equal(t, 1, i)

	// Test attempts less than 1 results in an error
	i = 0
	b := bytes.Buffer{}
	actualBytes, actualErr = readHttpWithRetries(func(url string) (int, string, io.ReadCloser, error) {
		return 200, "Status", ioutil.NopCloser(&b), nil
	}, 0, "hello", 0)
	assert.Error(t, actualErr)
	assert.Nil(t, actualBytes)
	assert.Equal(t, 0, i)

	// Test Success
	i = 0
	b = bytes.Buffer{}
	actualBytes, actualErr = readHttpWithRetries(func(url string) (int, string, io.ReadCloser, error) {
		assert.Equal(t, "hello", url)
		i++
		if i > 1 {
			return 200, "Status", ioutil.NopCloser(&b), nil
		}
		return 501, "Status", nil, nil
	}, 0, "hello", 3)
	assert.Nil(t, actualErr)
	assert.NotNil(t, actualBytes)
	assert.Equal(t, 2, i)
}
