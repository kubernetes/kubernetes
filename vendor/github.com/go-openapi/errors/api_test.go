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
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestServeError(t *testing.T) {
	// method not allowed wins
	var err error
	err = MethodNotAllowed("GET", []string{"POST", "PUT"})
	recorder := httptest.NewRecorder()
	ServeError(recorder, nil, err)
	assert.Equal(t, http.StatusMethodNotAllowed, recorder.Code)
	assert.Equal(t, "POST,PUT", recorder.Header().Get("Allow"))
	// assert.Equal(t, "application/json", recorder.Header().Get("content-type"))
	assert.Equal(t, `{"code":405,"message":"method GET is not allowed, but [POST,PUT] are"}`, recorder.Body.String())

	// renders status code from error when present
	err = NotFound("")
	recorder = httptest.NewRecorder()
	ServeError(recorder, nil, err)
	assert.Equal(t, http.StatusNotFound, recorder.Code)
	// assert.Equal(t, "application/json", recorder.Header().Get("content-type"))
	assert.Equal(t, `{"code":404,"message":"Not found"}`, recorder.Body.String())

	// defaults to internal server error
	err = fmt.Errorf("some error")
	recorder = httptest.NewRecorder()
	ServeError(recorder, nil, err)
	assert.Equal(t, http.StatusInternalServerError, recorder.Code)
	// assert.Equal(t, "application/json", recorder.Header().Get("content-type"))
	assert.Equal(t, `{"code":500,"message":"some error"}`, recorder.Body.String())
}

func TestAPIErrors(t *testing.T) {
	err := New(402, "this failed %s", "yada")
	assert.Error(t, err)
	assert.EqualValues(t, 402, err.Code())
	assert.EqualValues(t, "this failed yada", err.Error())

	err = NotFound("this failed %d", 1)
	assert.Error(t, err)
	assert.EqualValues(t, http.StatusNotFound, err.Code())
	assert.EqualValues(t, "this failed 1", err.Error())

	err = NotFound("")
	assert.Error(t, err)
	assert.EqualValues(t, http.StatusNotFound, err.Code())
	assert.EqualValues(t, "Not found", err.Error())

	err = NotImplemented("not implemented")
	assert.Error(t, err)
	assert.EqualValues(t, http.StatusNotImplemented, err.Code())
	assert.EqualValues(t, "not implemented", err.Error())

	err = MethodNotAllowed("GET", []string{"POST", "PUT"})
	assert.Error(t, err)
	assert.EqualValues(t, http.StatusMethodNotAllowed, err.Code())
	assert.EqualValues(t, "method GET is not allowed, but [POST,PUT] are", err.Error())

	err = InvalidContentType("application/saml", []string{"application/json", "application/x-yaml"})
	assert.Error(t, err)
	assert.EqualValues(t, http.StatusUnsupportedMediaType, err.Code())
	assert.EqualValues(t, "unsupported media type \"application/saml\", only [application/json application/x-yaml] are allowed", err.Error())

	err = InvalidResponseFormat("application/saml", []string{"application/json", "application/x-yaml"})
	assert.Error(t, err)
	assert.EqualValues(t, http.StatusNotAcceptable, err.Code())
	assert.EqualValues(t, "unsupported media type requested, only [application/json application/x-yaml] are available", err.Error())
}
