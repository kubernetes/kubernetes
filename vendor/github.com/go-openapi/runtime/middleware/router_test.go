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
	"sort"
	"strings"
	"testing"

	"github.com/go-openapi/analysis"
	"github.com/go-openapi/loads"
	"github.com/go-openapi/runtime/internal/testing/petstore"
	"github.com/go-openapi/runtime/middleware/untyped"
	"github.com/stretchr/testify/assert"
)

func terminator(rw http.ResponseWriter, r *http.Request) {
	rw.WriteHeader(http.StatusOK)
}

func TestRouterMiddleware(t *testing.T) {
	spec, api := petstore.NewAPI(t)
	context := NewContext(spec, api, nil)
	mw := newRouter(context, http.HandlerFunc(terminator))

	recorder := httptest.NewRecorder()
	request, _ := http.NewRequest("GET", "/api/pets", nil)

	mw.ServeHTTP(recorder, request)
	assert.Equal(t, 200, recorder.Code)

	recorder = httptest.NewRecorder()
	request, _ = http.NewRequest("DELETE", "/api/pets", nil)

	mw.ServeHTTP(recorder, request)
	assert.Equal(t, http.StatusMethodNotAllowed, recorder.Code)

	methods := strings.Split(recorder.Header().Get("Allow"), ",")
	sort.Sort(sort.StringSlice(methods))
	assert.Equal(t, "GET,POST", strings.Join(methods, ","))

	recorder = httptest.NewRecorder()
	request, _ = http.NewRequest("GET", "/nopets", nil)

	mw.ServeHTTP(recorder, request)
	assert.Equal(t, http.StatusNotFound, recorder.Code)

	spec, api = petstore.NewRootAPI(t)
	context = NewContext(spec, api, nil)
	mw = newRouter(context, http.HandlerFunc(terminator))

	recorder = httptest.NewRecorder()
	request, _ = http.NewRequest("GET", "/pets", nil)

	mw.ServeHTTP(recorder, request)
	assert.Equal(t, 200, recorder.Code)

	recorder = httptest.NewRecorder()
	request, _ = http.NewRequest("DELETE", "/pets", nil)

	mw.ServeHTTP(recorder, request)
	assert.Equal(t, http.StatusMethodNotAllowed, recorder.Code)

	methods = strings.Split(recorder.Header().Get("Allow"), ",")
	sort.Sort(sort.StringSlice(methods))
	assert.Equal(t, "GET,POST", strings.Join(methods, ","))

	recorder = httptest.NewRecorder()
	request, _ = http.NewRequest("GET", "/nopets", nil)

	mw.ServeHTTP(recorder, request)
	assert.Equal(t, http.StatusNotFound, recorder.Code)

}

func TestRouterBuilder(t *testing.T) {
	spec, api := petstore.NewAPI(t)
	analyzed := analysis.New(spec.Spec())

	assert.Len(t, analyzed.RequiredConsumes(), 3)
	assert.Len(t, analyzed.RequiredProduces(), 5)
	assert.Len(t, analyzed.OperationIDs(), 4)

	// context := NewContext(spec, api)
	builder := petAPIRouterBuilder(spec, api, analyzed)
	getRecords := builder.records["GET"]
	postRecords := builder.records["POST"]
	deleteRecords := builder.records["DELETE"]

	assert.Len(t, getRecords, 2)
	assert.Len(t, postRecords, 1)
	assert.Len(t, deleteRecords, 1)

	assert.Empty(t, builder.records["PATCH"])
	assert.Empty(t, builder.records["OPTIONS"])
	assert.Empty(t, builder.records["HEAD"])
	assert.Empty(t, builder.records["PUT"])

	rec := postRecords[0]
	assert.Equal(t, rec.Key, "/pets")
	val := rec.Value.(*routeEntry)
	assert.Len(t, val.Consumers, 1)
	assert.Len(t, val.Producers, 1)
	assert.Len(t, val.Consumes, 1)
	assert.Len(t, val.Produces, 1)

	assert.Len(t, val.Parameters, 1)

	recG := getRecords[0]
	assert.Equal(t, recG.Key, "/pets")
	valG := recG.Value.(*routeEntry)
	assert.Len(t, valG.Consumers, 2)
	assert.Len(t, valG.Producers, 4)
	assert.Len(t, valG.Consumes, 2)
	assert.Len(t, valG.Produces, 4)

	assert.Len(t, valG.Parameters, 2)
}

func TestRouterCanonicalBasePath(t *testing.T) {
	spec, api := petstore.NewAPI(t)
	spec.Spec().BasePath = "/api///"
	context := NewContext(spec, api, nil)
	mw := newRouter(context, http.HandlerFunc(terminator))

	recorder := httptest.NewRecorder()
	request, _ := http.NewRequest("GET", "/api/pets", nil)

	mw.ServeHTTP(recorder, request)
	assert.Equal(t, 200, recorder.Code)
}

func TestRouter_EscapedPath(t *testing.T) {
	spec, api := petstore.NewAPI(t)
	spec.Spec().BasePath = "/api/"
	context := NewContext(spec, api, nil)
	mw := newRouter(context, http.HandlerFunc(terminator))

	recorder := httptest.NewRecorder()
	request, _ := http.NewRequest("GET", "/api/pets/123", nil)

	mw.ServeHTTP(recorder, request)
	assert.Equal(t, 200, recorder.Code)

	recorder = httptest.NewRecorder()
	request, _ = http.NewRequest("GET", "/api/pets/abc%2Fdef", nil)

	mw.ServeHTTP(recorder, request)
	assert.Equal(t, 200, recorder.Code)
	ri, _ := context.RouteInfo(request)
	assert.Equal(t, "abc%2Fdef", ri.Params.Get("id"))
}

func TestRouterStruct(t *testing.T) {
	spec, api := petstore.NewAPI(t)
	router := DefaultRouter(spec, newRoutableUntypedAPI(spec, api, new(Context)))

	methods := router.OtherMethods("post", "/pets/{id}")
	assert.Len(t, methods, 2)

	entry, ok := router.Lookup("delete", "/pets/{id}")
	assert.True(t, ok)
	assert.NotNil(t, entry)
	assert.Len(t, entry.Params, 1)
	assert.Equal(t, "id", entry.Params[0].Name)

	_, ok = router.Lookup("delete", "/pets")
	assert.False(t, ok)

	_, ok = router.Lookup("post", "/no-pets")
	assert.False(t, ok)
}

func petAPIRouterBuilder(spec *loads.Document, api *untyped.API, analyzed *analysis.Spec) *defaultRouteBuilder {
	builder := newDefaultRouteBuilder(spec, newRoutableUntypedAPI(spec, api, new(Context)))
	builder.AddRoute("GET", "/pets", analyzed.AllPaths()["/pets"].Get)
	builder.AddRoute("POST", "/pets", analyzed.AllPaths()["/pets"].Post)
	builder.AddRoute("DELETE", "/pets/{id}", analyzed.AllPaths()["/pets/{id}"].Delete)
	builder.AddRoute("GET", "/pets/{id}", analyzed.AllPaths()["/pets/{id}"].Get)

	return builder
}
