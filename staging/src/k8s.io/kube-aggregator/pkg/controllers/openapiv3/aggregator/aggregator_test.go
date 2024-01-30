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
	"strings"
	"testing"

	"github.com/emicklei/go-restful/v3"
	"github.com/stretchr/testify/assert"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/endpoints/metrics"
	openapinamer "k8s.io/apiserver/pkg/endpoints/openapi"
	genericapiserver "k8s.io/apiserver/pkg/server"
	"k8s.io/apiserver/pkg/server/mux"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/metrics/testutil"
	v1 "k8s.io/kube-aggregator/pkg/apis/apiregistration/v1"
	openapicommon "k8s.io/kube-openapi/pkg/common"
	"k8s.io/kube-openapi/pkg/handler3"
	kubeopenapispec "k8s.io/kube-openapi/pkg/validation/spec"
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
	specProxier, err := BuildAndRegisterAggregator(downloader, genericapiserver.NewEmptyDelegate(), nil, nil, pathHandler)
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

	apiServiceNames := specProxier.GetAPIServiceNames()
	assert.ElementsMatch(t, []string{openAPIV2Converter, apiService.Name}, apiServiceNames)

	// Ensure that OpenAPI v3 for legacy APIService is removed.
	specProxier.RemoveAPIServiceSpec(apiService.Name)
	data = sendReq(t, serveHandler, "/openapi/v3")
	groupVersionList = handler3.OpenAPIV3Discovery{}
	if err := json.Unmarshal(data, &groupVersionList); err != nil {
		t.Fatal(err)
	}

	path, ok = groupVersionList.Paths["apis/group.example.com/v1"]
	if ok {
		t.Error("Expected group.example.com/v1 not to be in group version list")
	}
}

func TestV3APIService(t *testing.T) {
	downloader := Downloader{}

	pathHandler := mux.NewPathRecorderMux("aggregator_test")
	var serveHandler http.Handler = pathHandler
	specProxier, err := BuildAndRegisterAggregator(downloader, genericapiserver.NewEmptyDelegate(), nil, nil, pathHandler)
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

	apiServiceNames := specProxier.GetAPIServiceNames()
	assert.ElementsMatch(t, []string{openAPIV2Converter, apiService.Name}, apiServiceNames)
}

func TestV3RootAPIService(t *testing.T) {
	ws := new(restful.WebService)
	{
		ws.Path("/apis/apiregistration.k8s.io/v1")
		ws.Doc("API at/apis/apiregistration.k8s.io/v1 ")
		ws.Consumes("*/*")
		ws.Produces("application/json")
		ws.ApiVersion("apiregistration.k8s.io/v1")
		routeBuilder := ws.GET("apiservices").
			To(func(request *restful.Request, response *restful.Response) {}).
			Doc("list or watch objects of kind APIService").
			Operation("listAPIService").
			Produces("application/json").
			Returns(http.StatusOK, "OK", v1.APIService{}).
			Writes(v1.APIService{})
		ws.Route(routeBuilder)
	}
	openapiConfig := genericapiserver.DefaultOpenAPIV3Config(getTestAPIServiceOpenAPIDefinitions, openapinamer.NewDefinitionNamer(runtime.NewScheme()))

	downloader := Downloader{}
	goRestfulContainer := restful.NewContainer()
	goRestfulContainer.Add(ws)
	pathHandler := mux.NewPathRecorderMux("aggregator_test")
	var serveHandler http.Handler = pathHandler
	specProxier, err := BuildAndRegisterAggregator(downloader, genericapiserver.NewEmptyDelegate(), goRestfulContainer, openapiConfig, pathHandler)
	if err != nil {
		t.Error(err)
	}
	expectedSpecJSON := []byte(`{"openapi":"3.0.0","info":{"title":"Generic API Server"},"paths":{"/apis/apiregistration.k8s.io/v1/apiservices":{"get":{"tags":["apiregistration_v1"],"description":"list or watch objects of kind APIService","operationId":"listApiregistrationV1APIService","responses":{"200":{"description":"OK","content":{"application/json":{"schema":{"$ref":"#/components/schemas/io.k8s.kube-aggregator.pkg.apis.apiregistration.v1.APIService"}}}}}}}},"components":{"schemas":{"io.k8s.kube-aggregator.pkg.apis.apiregistration.v1.APIService":{"description":"APIService represents a server for a particular GroupVersion. Name must be \"version.group\".","type":"object"}}}}`)

	data := sendReq(t, serveHandler, "/openapi/v3")
	groupVersionList := handler3.OpenAPIV3Discovery{}
	if err := json.Unmarshal(data, &groupVersionList); err != nil {
		t.Fatal(err)
	}
	path, ok := groupVersionList.Paths["apis/apiregistration.k8s.io/v1"]
	if !ok {
		t.Error("Expected apiregistration.k8s.io/v1 to be in group version list")
	}
	gotSpecJSON := sendReq(t, serveHandler, path.ServerRelativeURL)
	if bytes.Compare(gotSpecJSON, expectedSpecJSON) != 0 {
		t.Errorf("Spec mismatch, expected %s, got %s", expectedSpecJSON, gotSpecJSON)
	}

	apiServiceNames := specProxier.GetAPIServiceNames()
	assert.ElementsMatch(t, []string{"k8s_internal_local_kube_aggregator_types", openAPIV2Converter}, apiServiceNames)
}

func TestOpenAPIRequestMetrics(t *testing.T) {
	metrics.Register()
	metrics.Reset()

	downloader := Downloader{}

	pathHandler := mux.NewPathRecorderMux("aggregator_metrics_test")
	var serveHandler http.Handler = pathHandler
	specProxier, err := BuildAndRegisterAggregator(downloader, genericapiserver.NewEmptyDelegate(), nil, nil, pathHandler)
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
	_, ok := groupVersionList.Paths["apis/group.example.com/v1"]
	if !ok {
		t.Error("Expected group.example.com/v1 to be in group version list")
	}

	// Metrics should be updated after requesting the root document.
	if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(`
# HELP apiserver_request_total [STABLE] Counter of apiserver requests broken out for each verb, dry run value, group, version, resource, scope, component, and HTTP response code.
# TYPE apiserver_request_total counter
apiserver_request_total{code="200",component="",dry_run="",group="",resource="",scope="",subresource="openapi/v3",verb="GET",version=""} 1
`), "apiserver_request_total"); err != nil {
		t.Fatal(err)
	}

	_ = sendReq(t, serveHandler, "/openapi/v3/apis/group.example.com/v1")

	// Metrics should be updated after requesting OpenAPI for a group version.
	if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(`
# HELP apiserver_request_total [STABLE] Counter of apiserver requests broken out for each verb, dry run value, group, version, resource, scope, component, and HTTP response code.
# TYPE apiserver_request_total counter
apiserver_request_total{code="200",component="",dry_run="",group="",resource="",scope="",subresource="openapi/v3",verb="GET",version=""} 1
apiserver_request_total{code="200",component="",dry_run="",group="",resource="",scope="",subresource="openapi/v3/",verb="GET",version=""} 1
`), "apiserver_request_total"); err != nil {
		t.Fatal(err)
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

func getTestAPIServiceOpenAPIDefinitions(_ openapicommon.ReferenceCallback) map[string]openapicommon.OpenAPIDefinition {
	return map[string]openapicommon.OpenAPIDefinition{
		"k8s.io/kube-aggregator/pkg/apis/apiregistration/v1.APIService": buildTestAPIServiceOpenAPIDefinition(),
	}
}

func buildTestAPIServiceOpenAPIDefinition() openapicommon.OpenAPIDefinition {
	return openapicommon.OpenAPIDefinition{
		Schema: kubeopenapispec.Schema{
			SchemaProps: kubeopenapispec.SchemaProps{
				Description: "APIService represents a server for a particular GroupVersion. Name must be \"version.group\".",
				Type:        []string{"object"},
			},
		},
	}
}
