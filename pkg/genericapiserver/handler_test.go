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

package genericapiserver

import (
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestIntercept404ResponseWriterCode(t *testing.T) {
	w := httptest.NewRecorder()
	aw := NewIntercept404ResponseWriter(w)

	assert.Equal(t, 0, aw.Code())

	aw.WriteHeader(42)
	assert.Equal(t, 42, aw.Code())
	assert.Equal(t, 42, w.Code)
}

func TestIntercept404ResponseWriterDirectWrite(t *testing.T) {
	w := httptest.NewRecorder()
	aw := NewIntercept404ResponseWriter(w)

	written, err := aw.Write([]byte("foo"))
	assert.Equal(t, 200, aw.Code())
	assert.NoError(t, err)
	assert.Equal(t, 3, written)
}

func TestIntercept404ResponseWriterHeader(t *testing.T) {
	w := httptest.NewRecorder()
	w.Header().Set("foo", "bar")
	aw := NewIntercept404ResponseWriter(w)

	assert.Equal(t, "bar", aw.Header().Get("foo"))

	aw.Header().Set("foo", "notbar")
	assert.Equal(t, "notbar", aw.Header().Get("foo"))
	assert.Equal(t, "bar", w.Header().Get("foo"))

	aw.WriteHeader(200)
	assert.Equal(t, "notbar", w.Header().Get("foo"))

	aw.Header().Set("foo", "notnotbar")
	assert.Equal(t, "notnotbar", w.Header().Get("foo"))
}

func TestIntercept404ResponseWriter404(t *testing.T) {
	w := httptest.NewRecorder()
	w.Header().Set("foo", "bar")
	aw := NewIntercept404ResponseWriter(w)

	assert.Equal(t, 0, w.Body.Len())
	aw.WriteHeader(404)

	aw.Header().Set("foo", "notbar")
	assert.Equal(t, "bar", w.Header().Get("foo"))

	written, err := aw.Write([]byte("foo"))
	assert.NoError(t, err)
	assert.Equal(t, 3, written, 3)
	assert.Equal(t, 0, w.Body.Len())
}

func TestMergeHandlers(t *testing.T) {
	type Test struct {
		codeA, codeB int
		expectedCode int
		expectedBody string
	}
	for i, test := range []Test{
		{200, 200, 200, "a"},
		{200, 404, 200, "a"},
		{404, 200, 200, "b"},
		{404, 404, 404, "b"},
	} {
		a := http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
			w.WriteHeader(test.codeA)
			w.Write([]byte("a"))
		})
		b := http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
			w.WriteHeader(test.codeB)
			w.Write([]byte("b"))
		})

		handler := MergeHandlers(a, b)
		var r io.Reader
		w := httptest.NewRecorder()
		req, _ := http.NewRequest("GET", "/", r)
		handler.ServeHTTP(w, req)

		assert.Equal(t, test.expectedCode, w.Code, "%d: expected code %d, got %d", i, test.expectedCode, w.Code)
		assert.Equal(t, test.expectedBody, w.Body.String(), "%d: expected body %q, got %q", i, test.expectedBody, w.Body.String())
	}
}
