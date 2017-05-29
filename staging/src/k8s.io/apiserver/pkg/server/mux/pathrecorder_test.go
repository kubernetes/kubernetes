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
	c := NewPathRecorderMux("test")
	c.UnlistedHandleFunc("/secret", func(http.ResponseWriter, *http.Request) {})
	c.HandleFunc("/nonswagger", func(http.ResponseWriter, *http.Request) {})
	assert.NotContains(t, c.ListedPaths(), "/secret")
	assert.Contains(t, c.ListedPaths(), "/nonswagger")
}

func TestUnregisterHandlers(t *testing.T) {
	first := 0
	second := 0

	c := NewPathRecorderMux("test")
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

func TestPrefixHandlers(t *testing.T) {
	c := NewPathRecorderMux("test")
	s := httptest.NewServer(c)
	defer s.Close()

	secretPrefixCount := 0
	c.UnlistedHandlePrefix("/secretPrefix/", http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		secretPrefixCount = secretPrefixCount + 1
	}))
	publicPrefixCount := 0
	c.HandlePrefix("/publicPrefix/", http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		publicPrefixCount = publicPrefixCount + 1
	}))
	precisePrefixCount := 0
	c.HandlePrefix("/publicPrefix/but-more-precise/", http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		precisePrefixCount = precisePrefixCount + 1
	}))
	exactMatchCount := 0
	c.Handle("/publicPrefix/exactmatch", http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		exactMatchCount = exactMatchCount + 1
	}))
	slashMatchCount := 0
	c.Handle("/otherPublic/exactmatchslash/", http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		slashMatchCount = slashMatchCount + 1
	}))
	fallThroughCount := 0
	c.NotFoundHandler(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		fallThroughCount = fallThroughCount + 1
	}))

	assert.NotContains(t, c.ListedPaths(), "/secretPrefix/")
	assert.Contains(t, c.ListedPaths(), "/publicPrefix/")

	resp, _ := http.Get(s.URL + "/fallthrough")
	assert.Equal(t, 1, fallThroughCount)
	assert.Equal(t, resp.StatusCode, http.StatusOK)
	resp, _ = http.Get(s.URL + "/publicPrefix")
	assert.Equal(t, 2, fallThroughCount)
	assert.Equal(t, resp.StatusCode, http.StatusOK)

	http.Get(s.URL + "/publicPrefix/")
	assert.Equal(t, 1, publicPrefixCount)
	http.Get(s.URL + "/publicPrefix/something")
	assert.Equal(t, 2, publicPrefixCount)
	http.Get(s.URL + "/publicPrefix/but-more-precise")
	assert.Equal(t, 3, publicPrefixCount)
	http.Get(s.URL + "/publicPrefix/but-more-precise/")
	assert.Equal(t, 1, precisePrefixCount)
	http.Get(s.URL + "/publicPrefix/but-more-precise/more-stuff")
	assert.Equal(t, 2, precisePrefixCount)

	http.Get(s.URL + "/publicPrefix/exactmatch")
	assert.Equal(t, 1, exactMatchCount)
	http.Get(s.URL + "/publicPrefix/exactmatch/")
	assert.Equal(t, 4, publicPrefixCount)
	http.Get(s.URL + "/otherPublic/exactmatchslash")
	assert.Equal(t, 3, fallThroughCount)
	http.Get(s.URL + "/otherPublic/exactmatchslash/")
	assert.Equal(t, 1, slashMatchCount)

	http.Get(s.URL + "/secretPrefix/")
	assert.Equal(t, 1, secretPrefixCount)
	http.Get(s.URL + "/secretPrefix/something")
	assert.Equal(t, 2, secretPrefixCount)
}
