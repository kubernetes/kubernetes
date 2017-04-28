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

package mux

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestSecretHandlers(t *testing.T) {
	c := NewPathRecorderMux()
	c.UnlistedHandleFunc("/secret", func(http.ResponseWriter, *http.Request) {})
	c.HandleFunc("/nonswagger", func(http.ResponseWriter, *http.Request) {})
	assert.NotContains(t, c.ListedPaths(), "/secret")
	assert.Contains(t, c.ListedPaths(), "/nonswagger")
}

func TestUnregisterHandlers(t *testing.T) {
	first := 0
	second := 0

	c := NewPathRecorderMux()
	s := httptest.NewServer(c)
	defer s.Close()

	c.UnlistedHandleFunc("/secret", func(http.ResponseWriter, *http.Request) {})
	c.HandleFunc("/nonswagger", func(http.ResponseWriter, *http.Request) {
		first = first + 1
	})
	assert.NotContains(t, c.ListedPaths(), "/secret")
	assert.Contains(t, c.ListedPaths(), "/nonswagger")

	resp, _ := http.Get(s.URL + "/nonswagger")
	assert.Equal(t, first, 1)
	assert.Equal(t, resp.StatusCode, http.StatusOK)

	c.Unregister("/nonswagger")
	assert.NotContains(t, c.ListedPaths(), "/nonswagger")

	resp, _ = http.Get(s.URL + "/nonswagger")
	assert.Equal(t, first, 1)
	assert.Equal(t, resp.StatusCode, http.StatusNotFound)

	c.HandleFunc("/nonswagger", func(http.ResponseWriter, *http.Request) {
		second = second + 1
	})
	assert.Contains(t, c.ListedPaths(), "/nonswagger")
	resp, _ = http.Get(s.URL + "/nonswagger")
	assert.Equal(t, first, 1)
	assert.Equal(t, second, 1)
	assert.Equal(t, resp.StatusCode, http.StatusOK)
}
