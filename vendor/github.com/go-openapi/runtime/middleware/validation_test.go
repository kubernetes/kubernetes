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
	"bytes"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/go-openapi/errors"
	"github.com/go-openapi/runtime"
	"github.com/go-openapi/runtime/internal/testing/petstore"
	"github.com/stretchr/testify/assert"
)

func TestContentTypeValidation(t *testing.T) {
	spec, api := petstore.NewAPI(t)
	context := NewContext(spec, api, nil)
	context.router = DefaultRouter(spec, context.api)
	mw := newValidation(context, http.HandlerFunc(terminator))

	recorder := httptest.NewRecorder()
	request, _ := http.NewRequest("GET", "/pets", nil)
	request.Header.Add("Accept", "*/*")
	mw.ServeHTTP(recorder, request)
	assert.Equal(t, 200, recorder.Code)

	recorder = httptest.NewRecorder()
	request, _ = http.NewRequest("POST", "/pets", nil)
	request.Header.Add("content-type", "application(")
	request.ContentLength = 1

	mw.ServeHTTP(recorder, request)
	assert.Equal(t, 400, recorder.Code)
	assert.Equal(t, "application/json", recorder.Header().Get("content-type"))

	recorder = httptest.NewRecorder()
	request, _ = http.NewRequest("POST", "/pets", nil)
	request.Header.Add("Accept", "application/json")
	request.Header.Add("content-type", "text/html")
	request.ContentLength = 1

	mw.ServeHTTP(recorder, request)
	assert.Equal(t, 415, recorder.Code)
	assert.Equal(t, "application/json", recorder.Header().Get("content-type"))

	recorder = httptest.NewRecorder()
	request, _ = http.NewRequest("POST", "/pets", nil)
	request.Header.Add("Accept", "application/json")
	request.Header.Add("content-type", "text/html")
	request.TransferEncoding = []string{"chunked"}

	mw.ServeHTTP(recorder, request)
	assert.Equal(t, 415, recorder.Code)
	assert.Equal(t, "application/json", recorder.Header().Get("content-type"))

	recorder = httptest.NewRecorder()
	request, _ = http.NewRequest("POST", "/pets", nil)
	request.Header.Add("Accept", "application/json")
	request.Header.Add("content-type", "text/html")

	mw.ServeHTTP(recorder, request)
	assert.Equal(t, 422, recorder.Code)
	assert.Equal(t, "application/json", recorder.Header().Get("content-type"))
}

func TestResponseFormatValidation(t *testing.T) {
	spec, api := petstore.NewAPI(t)
	context := NewContext(spec, api, nil)
	context.router = DefaultRouter(spec, context.api)
	mw := newValidation(context, http.HandlerFunc(terminator))

	recorder := httptest.NewRecorder()
	request, _ := http.NewRequest("POST", "/pets", bytes.NewBuffer([]byte(`name: Dog`)))
	request.Header.Set(runtime.HeaderContentType, "application/x-yaml")
	request.Header.Set(runtime.HeaderAccept, "application/x-yaml")

	mw.ServeHTTP(recorder, request)
	assert.Equal(t, 200, recorder.Code, recorder.Body.String())

	recorder = httptest.NewRecorder()
	request, _ = http.NewRequest("POST", "/pets", bytes.NewBuffer([]byte(`name: Dog`)))
	request.Header.Set(runtime.HeaderContentType, "application/x-yaml")
	request.Header.Set(runtime.HeaderAccept, "application/sml")

	mw.ServeHTTP(recorder, request)
	assert.Equal(t, http.StatusNotAcceptable, recorder.Code)
}

func TestValidateContentType(t *testing.T) {
	data := []struct {
		hdr     string
		allowed []string
		err     *errors.Validation
	}{
		{"application/json", []string{"application/json"}, nil},
		{"application/json", []string{"application/x-yaml", "text/html"}, errors.InvalidContentType("application/json", []string{"application/x-yaml", "text/html"})},
		{"text/html; charset=utf-8", []string{"text/html"}, nil},
		{"text/html;charset=utf-8", []string{"text/html"}, nil},
		{"", []string{"application/json"}, errors.InvalidContentType("", []string{"application/json"})},
		{"text/html;           charset=utf-8", []string{"application/json"}, errors.InvalidContentType("text/html;           charset=utf-8", []string{"application/json"})},
		{"application(", []string{"application/json"}, errors.InvalidContentType("application(", []string{"application/json"})},
		{"application/json;char*", []string{"application/json"}, errors.InvalidContentType("application/json;char*", []string{"application/json"})},
	}

	for _, v := range data {
		err := validateContentType(v.allowed, v.hdr)
		if v.err == nil {
			assert.NoError(t, err, "input: %q", v.hdr)
		} else {
			assert.Error(t, err, "input: %q", v.hdr)
			assert.IsType(t, &errors.Validation{}, err, "input: %q", v.hdr)
			assert.Equal(t, v.err.Error(), err.Error(), "input: %q", v.hdr)
			assert.EqualValues(t, http.StatusUnsupportedMediaType, err.(*errors.Validation).Code())
		}
	}
}
