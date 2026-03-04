/*
Copyright 2021 The Kubernetes Authors.

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

package testing

import (
	"encoding/json"
	"io"
	"io/fs"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"k8s.io/kube-openapi/pkg/handler3"
	"k8s.io/kube-openapi/pkg/spec3"
)

type FakeOpenAPIServer struct {
	HttpServer      *httptest.Server
	ServedDocuments map[string]*spec3.OpenAPI
	RequestCounters map[string]int
}

// Creates a mock OpenAPIV3 server as it would be on a standing kubernetes
// API server.
//
// specsPath - Give a path to some test data organized so that each GroupVersion
// has its own OpenAPI V3 JSON file.
//
//	i.e. apps/v1beta1 is stored in <specsPath>/apps/v1beta1.json
func NewFakeOpenAPIV3Server(specsPath string) (*FakeOpenAPIServer, error) {
	mux := &testMux{
		counts: map[string]int{},
	}
	server := httptest.NewServer(mux)

	openAPIVersionedService := handler3.NewOpenAPIService()
	err := openAPIVersionedService.RegisterOpenAPIV3VersionedService("/openapi/v3", mux)
	if err != nil {
		return nil, err
	}

	grouped := make(map[string][]byte)
	var testV3Specs = make(map[string]*spec3.OpenAPI)

	addSpec := func(path string) {
		file, err := os.Open(path)
		if err != nil {
			panic(err)
		}

		defer file.Close()
		vals, err := io.ReadAll(file)
		if err != nil {
			panic(err)
		}

		rel, err := filepath.Rel(specsPath, path)
		if err == nil {
			grouped[rel[:(len(rel)-len(filepath.Ext(rel)))]] = vals
		}
	}

	filepath.WalkDir(specsPath, func(path string, d fs.DirEntry, err error) error {
		if filepath.Ext(path) != ".json" || d.IsDir() {
			return nil
		}

		addSpec(path)
		return nil
	})

	for gv, jsonSpec := range grouped {
		spec := &spec3.OpenAPI{}
		err = json.Unmarshal(jsonSpec, spec)
		if err != nil {
			return nil, err
		}

		testV3Specs[gv] = spec
		openAPIVersionedService.UpdateGroupVersion(gv, spec)
	}

	return &FakeOpenAPIServer{
		HttpServer:      server,
		ServedDocuments: testV3Specs,
		RequestCounters: mux.counts,
	}, nil
}

////////////////////////////////////////////////////////////////////////////////
// Tiny Test HTTP Mux
////////////////////////////////////////////////////////////////////////////////
// Implements the mux interface used by handler3 for registering the OpenAPI
//	handlers

type testMux struct {
	lock      sync.Mutex
	prefixMap map[string]http.Handler
	pathMap   map[string]http.Handler
	counts    map[string]int
}

func (t *testMux) Handle(path string, handler http.Handler) {
	t.lock.Lock()
	defer t.lock.Unlock()

	if t.pathMap == nil {
		t.pathMap = make(map[string]http.Handler)
	}
	t.pathMap[path] = handler
}

func (t *testMux) HandlePrefix(path string, handler http.Handler) {
	t.lock.Lock()
	defer t.lock.Unlock()

	if t.prefixMap == nil {
		t.prefixMap = make(map[string]http.Handler)
	}
	t.prefixMap[path] = handler
}

func (t *testMux) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	t.lock.Lock()
	defer t.lock.Unlock()

	if t.counts == nil {
		t.counts = make(map[string]int)
	}

	if val, exists := t.counts[req.URL.Path]; exists {
		t.counts[req.URL.Path] = val + 1
	} else {
		t.counts[req.URL.Path] = 1
	}

	if handler, ok := t.pathMap[req.URL.Path]; ok {
		handler.ServeHTTP(w, req)
		return
	}

	for k, v := range t.prefixMap {
		if strings.HasPrefix(req.URL.Path, k) {
			v.ServeHTTP(w, req)
			return
		}
	}

	w.WriteHeader(http.StatusNotFound)
}
