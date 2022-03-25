/*
Copyright 2022 The Kubernetes Authors.

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

package aggregator

import (
	"bytes"
	"encoding/json"
	"net/http"
	"testing"

	genericapiserver "k8s.io/apiserver/pkg/server"
	"k8s.io/apiserver/pkg/server/mux"
	v1 "k8s.io/kube-aggregator/pkg/apis/apiregistration/v1"
	"k8s.io/kube-openapi/pkg/handler3"
)

type testV3APIService struct {
	etag string
	data []byte
}

var _ http.Handler = testV3APIService{}

func (h testV3APIService) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	// Create an APIService with a handler for one group/version
	if r.URL.Path == "/openapi/v3" {
		group := &handler3.OpenAPIV3Discovery{
			Paths: map[string]handler3.OpenAPIV3DiscoveryGroupVersion{
				"apis/group.example.com/v1": {
					ServerRelativeURL: "/openapi/v3/apis/group.example.com/v1?hash=" + h.etag,
				},
			},
		}

		j, _ := json.Marshal(group)
		w.Write(j)
		return
	}

	if r.URL.Path == "/openapi/v3/apis/group.example.com/v1" {
		if len(h.etag) > 0 {
			w.Header().Add("Etag", h.etag)
		}
		ifNoneMatches := r.Header["If-None-Match"]
		for _, match := range ifNoneMatches {
			if match == h.etag {
				w.WriteHeader(http.StatusNotModified)
				return
			}
		}
		w.Write(h.data)
	}
}

type testV2APIService struct{}

var _ http.Handler = testV2APIService{}

func (h testV2APIService) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	// Create an APIService with a handler for one group/version
	if r.URL.Path == "/openapi/v2" {
		w.Write([]byte(`{"swagger":"2.0","info":{"title":"Kubernetes","version":"unversioned"}}`))
		return
	}
	w.WriteHeader(404)
}

func TestV2APIService(t *testing.T) {
	downloader := Downloader{}
	pathHandler := mux.NewPathRecorderMux("aggregator_test")
	var serveHandler http.Handler = pathHandler
	specProxier, err := BuildAndRegisterAggregator(downloader, genericapiserver.NewEmptyDelegate(), pathHandler)
	if err != nil {
		t.Error(err)
	}
	handler := testV2APIService{}
	apiService := &v1.APIService{
		Spec: v1.APIServiceSpec{
			Group:   "group.example.com",
			Version: "v1",
		},
	}
	apiService.Name = "v1.group.example.com"
	specProxier.AddUpdateAPIService(handler, apiService)
	specProxier.UpdateAPIServiceSpec("v1.group.example.com")

	data := sendReq(t, serveHandler, "/openapi/v3")
	groupVersionList := handler3.OpenAPIV3Discovery{}
	if err := json.Unmarshal(data, &groupVersionList); err != nil {
		t.Fatal(err)
	}

	// A legacy APIService will not publish OpenAPI V3
	// Ensure that we can still aggregate its V2 spec and convert it to V3.
	path, ok := groupVersionList.Paths["apis/group.example.com/v1"]
	if !ok {
		t.Error("Expected group.example.com/v1 to be in group version list")
	}
	gotSpecJSON := sendReq(t, serveHandler, path.ServerRelativeURL)

	expectedV3Bytes := []byte(`{"openapi":"3.0.0","info":{"title":"Kubernetes","version":"unversioned"},"components":{}}`)

	if bytes.Compare(gotSpecJSON, expectedV3Bytes) != 0 {
		t.Errorf("Spec mismatch, expected %s, got %s", expectedV3Bytes, gotSpecJSON)
	}
}

func TestV3APIService(t *testing.T) {
	downloader := Downloader{}

	pathHandler := mux.NewPathRecorderMux("aggregator_test")
	var serveHandler http.Handler = pathHandler
	specProxier, err := BuildAndRegisterAggregator(downloader, genericapiserver.NewEmptyDelegate(), pathHandler)
	if err != nil {
		t.Error(err)
	}
	specJSON := []byte(`{"openapi":"3.0.0","info":{"title":"Kubernetes","version":"unversioned"}}`)
	handler := testV3APIService{
		etag: "6E8F849B434D4B98A569B9D7718876E9-356ECAB19D7FBE1336BABB1E70F8F3025050DE218BE78256BE81620681CFC9A268508E542B8B55974E17B2184BBFC8FFFAA577E51BE195D32B3CA2547818ABE4",
		data: specJSON,
	}
	apiService := &v1.APIService{
		Spec: v1.APIServiceSpec{
			Group:   "group.example.com",
			Version: "v1",
		},
	}
	apiService.Name = "v1.group.example.com"
	specProxier.AddUpdateAPIService(handler, apiService)
	specProxier.UpdateAPIServiceSpec("v1.group.example.com")

	data := sendReq(t, serveHandler, "/openapi/v3")
	groupVersionList := handler3.OpenAPIV3Discovery{}
	if err := json.Unmarshal(data, &groupVersionList); err != nil {
		t.Fatal(err)
	}
	path, ok := groupVersionList.Paths["apis/group.example.com/v1"]
	if !ok {
		t.Error("Expected group.example.com/v1 to be in group version list")
	}
	gotSpecJSON := sendReq(t, serveHandler, path.ServerRelativeURL)
	if bytes.Compare(gotSpecJSON, specJSON) != 0 {
		t.Errorf("Spec mismatch, expected %s, got %s", specJSON, gotSpecJSON)
	}
}

func sendReq(t *testing.T, handler http.Handler, path string) []byte {
	req, err := http.NewRequest("GET", path, nil)
	if err != nil {
		t.Fatal(err)
	}
	writer := newInMemoryResponseWriter()
	handler.ServeHTTP(writer, req)
	return writer.data
}
