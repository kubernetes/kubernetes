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

package simplepetstore

import (
	"bytes"
	"net/http/httptest"
	"testing"

	"github.com/go-openapi/runtime"
	"github.com/stretchr/testify/assert"
)

func TestSimplePetstoreSpec(t *testing.T) {
	handler, _ := NewPetstore()
	// Serves swagger spec document
	r, _ := runtime.JSONRequest("GET", "/swagger.json", nil)
	rw := httptest.NewRecorder()
	handler.ServeHTTP(rw, r)
	assert.Equal(t, 200, rw.Code)
	assert.Equal(t, swaggerJSON, rw.Body.String())
}

func TestSimplePetstoreAllPets(t *testing.T) {
	handler, _ := NewPetstore()
	// Serves swagger spec document
	r, _ := runtime.JSONRequest("GET", "/api/pets", nil)
	rw := httptest.NewRecorder()
	handler.ServeHTTP(rw, r)
	assert.Equal(t, 200, rw.Code)
	assert.Equal(t, "[{\"id\":1,\"name\":\"Dog\",\"status\":\"available\"},{\"id\":2,\"name\":\"Cat\",\"status\":\"pending\"}]\n", rw.Body.String())
}

func TestSimplePetstorePetByID(t *testing.T) {
	handler, _ := NewPetstore()

	// Serves swagger spec document
	r, _ := runtime.JSONRequest("GET", "/api/pets/1", nil)
	rw := httptest.NewRecorder()
	handler.ServeHTTP(rw, r)
	assert.Equal(t, 200, rw.Code)
	assert.Equal(t, "{\"id\":1,\"name\":\"Dog\",\"status\":\"available\"}\n", rw.Body.String())
}

func TestSimplePetstoreAddPet(t *testing.T) {
	handler, _ := NewPetstore()
	// Serves swagger spec document
	r, _ := runtime.JSONRequest("POST", "/api/pets", bytes.NewBuffer([]byte(`{"name": "Fish","status": "available"}`)))
	rw := httptest.NewRecorder()
	handler.ServeHTTP(rw, r)
	assert.Equal(t, 200, rw.Code)
	assert.Equal(t, "{\"id\":3,\"name\":\"Fish\",\"status\":\"available\"}\n", rw.Body.String())
}

func TestSimplePetstoreDeletePet(t *testing.T) {
	handler, _ := NewPetstore()
	// Serves swagger spec document
	r, _ := runtime.JSONRequest("DELETE", "/api/pets/1", nil)
	rw := httptest.NewRecorder()
	handler.ServeHTTP(rw, r)
	assert.Equal(t, 204, rw.Code)
	assert.Equal(t, "", rw.Body.String())

	r, _ = runtime.JSONRequest("GET", "/api/pets/1", nil)
	rw = httptest.NewRecorder()
	handler.ServeHTTP(rw, r)
	assert.Equal(t, 404, rw.Code)
	assert.Equal(t, "{\"code\":404,\"message\":\"not found: pet 1\"}", rw.Body.String())
}
