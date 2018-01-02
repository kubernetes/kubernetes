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
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/go-openapi/runtime"
	"github.com/go-openapi/runtime/internal/testing/petstore"
	"github.com/stretchr/testify/assert"
)

func TestServeSpecMiddleware(t *testing.T) {
	spec, api := petstore.NewAPI(t)
	ctx := NewContext(spec, api, nil)

	handler := specMiddleware(ctx, nil)
	// serves spec
	request, _ := http.NewRequest("GET", "/swagger.json", nil)
	request.Header.Add(runtime.HeaderContentType, runtime.JSONMime)
	recorder := httptest.NewRecorder()
	handler.ServeHTTP(recorder, request)
	assert.Equal(t, 200, recorder.Code)

	// returns 404 when no next handler
	request, _ = http.NewRequest("GET", "/api/pets", nil)
	request.Header.Add(runtime.HeaderContentType, runtime.JSONMime)
	recorder = httptest.NewRecorder()
	handler.ServeHTTP(recorder, request)
	assert.Equal(t, 404, recorder.Code)

	// forwards to next handler for other url
	handler = specMiddleware(ctx, http.HandlerFunc(func(rw http.ResponseWriter, r *http.Request) {
		rw.WriteHeader(http.StatusOK)
	}))
	request, _ = http.NewRequest("GET", "/api/pets", nil)
	request.Header.Add(runtime.HeaderContentType, runtime.JSONMime)
	recorder = httptest.NewRecorder()
	handler.ServeHTTP(recorder, request)
	assert.Equal(t, 200, recorder.Code)

}
