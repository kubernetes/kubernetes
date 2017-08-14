/*
Copyright 2017 The Kubernetes Authors.

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

package apiserver

import (
	"encoding/json"
	"fmt"
	"net/http"
	"sort"
	"time"

	"github.com/emicklei/go-restful"
	"github.com/go-openapi/spec"

	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/kube-aggregator/pkg/apis/apiregistration"
	"k8s.io/kube-openapi/pkg/aggregator"
	"k8s.io/kube-openapi/pkg/builder"
	"k8s.io/kube-openapi/pkg/common"
	"k8s.io/kube-openapi/pkg/handler"
)

const (
	aggregatorUser      = "system:aggregator"
	specDownloadTimeout = 60 * time.Second
)

type openAPIAggregator struct {
	// Map of API Services' OpenAPI specs by their name
	openAPISpecs map[string]*openAPISpecInfo

	// provided for dynamic OpenAPI spec
	openAPIService *handler.OpenAPIService

	// Aggregator's OpenAPI spec (holds apiregistration group).
	aggregatorOpenAPISpec *spec.Swagger

	// Local (in process) delegate's OpenAPI spec.
	inProcessDelegatesOpenAPISpec *spec.Swagger

	contextMapper request.RequestContextMapper
}

func buildAndRegisterOpenAPIAggregator(delegateHandler http.Handler, webServices []*restful.WebService, config *common.Config, pathHandler common.PathHandler, contextMapper request.RequestContextMapper) (s *openAPIAggregator, err error) {
	s = &openAPIAggregator{
		openAPISpecs:  map[string]*openAPISpecInfo{},
		contextMapper: contextMapper,
	}

	// Get Local delegate's Spec
	s.inProcessDelegatesOpenAPISpec, err = s.downloadOpenAPISpec(delegateHandler)
	if err != nil {
		return nil, err
	}

	// Build Aggregator's spec
	s.aggregatorOpenAPISpec, err = builder.BuildOpenAPISpec(
		webServices, config)
	if err != nil {
		return nil, err
	}
	// Remove any non-API endpoints from aggregator's spec. aggregatorOpenAPISpec
	// is the source of truth for all non-api endpoints.
	aggregator.FilterSpecByPaths(s.aggregatorOpenAPISpec, []string{"/apis/"})

	// Build initial spec to serve.
	specToServe, err := s.buildOpenAPISpec()
	if err != nil {
		return nil, err
	}

	// Install handler
	s.openAPIService, err = handler.RegisterOpenAPIService(
		specToServe, "/swagger.json", pathHandler)
	if err != nil {
		return nil, err
	}

	return s, nil
}

// openAPISpecInfo is used to store OpenAPI spec with its priority.
// It can be used to sort specs with their priorities.
type openAPISpecInfo struct {
	apiService apiregistration.APIService
	spec       *spec.Swagger
}

// byPriority can be used in sort.Sort to sort specs with their priorities.
type byPriority struct {
	specs           []openAPISpecInfo
	groupPriorities map[string]int32
}

func (a byPriority) Len() int      { return len(a.specs) }
func (a byPriority) Swap(i, j int) { a.specs[i], a.specs[j] = a.specs[j], a.specs[i] }
func (a byPriority) Less(i, j int) bool {
	var iPriority, jPriority int32
	if a.specs[i].apiService.Spec.Group == a.specs[j].apiService.Spec.Group {
		iPriority = a.specs[i].apiService.Spec.VersionPriority
		jPriority = a.specs[i].apiService.Spec.VersionPriority
	} else {
		iPriority = a.groupPriorities[a.specs[i].apiService.Spec.Group]
		jPriority = a.groupPriorities[a.specs[j].apiService.Spec.Group]
	}
	if iPriority != jPriority {
		// Sort by priority, higher first
		return iPriority > jPriority
	}
	// Sort by service name.
	return a.specs[i].apiService.Name < a.specs[j].apiService.Name
}

func sortByPriority(specs []openAPISpecInfo) {
	b := byPriority{
		specs:           specs,
		groupPriorities: map[string]int32{},
	}
	for _, spec := range specs {
		if pr, found := b.groupPriorities[spec.apiService.Spec.Group]; !found || spec.apiService.Spec.GroupPriorityMinimum > pr {
			b.groupPriorities[spec.apiService.Spec.Group] = spec.apiService.Spec.GroupPriorityMinimum
		}
	}
	sort.Sort(b)
}

// buildOpenAPISpec aggregates all OpenAPI specs.  It is not thread-safe.
func (s *openAPIAggregator) buildOpenAPISpec() (specToReturn *spec.Swagger, err error) {
	specToReturn, err = aggregator.CloneSpec(s.inProcessDelegatesOpenAPISpec)
	if err != nil {
		return nil, err
	}
	if err := aggregator.MergeSpecs(specToReturn, s.aggregatorOpenAPISpec); err != nil {
		return nil, fmt.Errorf("cannot merge local delegate spec with aggregator spec: %s", err.Error())
	}
	specs := []openAPISpecInfo{}
	for _, specInfo := range s.openAPISpecs {
		specs = append(specs, openAPISpecInfo{specInfo.apiService, specInfo.spec})
	}
	sortByPriority(specs)
	for _, specInfo := range specs {
		if err := aggregator.MergeSpecs(specToReturn, specInfo.spec); err != nil {
			return nil, err
		}
	}
	return specToReturn, nil
}

// updateOpenAPISpec aggregates all OpenAPI specs.  It is not thread-safe.
func (s *openAPIAggregator) updateOpenAPISpec() error {
	if s.openAPIService == nil {
		return nil
	}
	specToServe, err := s.buildOpenAPISpec()
	if err != nil {
		return err
	}
	return s.openAPIService.UpdateSpec(specToServe)
}

// inMemoryResponseWriter is a http.Writer that keep the response in memory.
type inMemoryResponseWriter struct {
	header   http.Header
	respCode int
	data     []byte
}

func newInMemoryResponseWriter() *inMemoryResponseWriter {
	return &inMemoryResponseWriter{header: http.Header{}}
}

func (r *inMemoryResponseWriter) Header() http.Header {
	return r.header
}

func (r *inMemoryResponseWriter) WriteHeader(code int) {
	r.respCode = code
}

func (r *inMemoryResponseWriter) Write(in []byte) (int, error) {
	r.data = append(r.data, in...)
	return len(in), nil
}

func (r *inMemoryResponseWriter) String() string {
	s := fmt.Sprintf("ResponseCode: %d", r.respCode)
	if r.data != nil {
		s += fmt.Sprintf(", Body: %s", string(r.data))
	}
	if r.header != nil {
		s += fmt.Sprintf(", Header: %s", r.header)
	}
	return s
}

func (s *openAPIAggregator) handlerWithUser(handler http.Handler, info user.Info) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		if ctx, ok := s.contextMapper.Get(req); ok {
			s.contextMapper.Update(req, request.WithUser(ctx, info))
		}
		handler.ServeHTTP(w, req)
	})
}

// downloadOpenAPISpec downloads openAPI spec from /swagger.json endpoint of the given handler.
func (s *openAPIAggregator) downloadOpenAPISpec(handler http.Handler) (*spec.Swagger, error) {
	handler = s.handlerWithUser(handler, &user.DefaultInfo{Name: aggregatorUser})
	handler = request.WithRequestContext(handler, s.contextMapper)
	handler = http.TimeoutHandler(handler, specDownloadTimeout, "request timed out")

	req, err := http.NewRequest("GET", "/swagger.json", nil)
	if err != nil {
		return nil, err
	}
	writer := newInMemoryResponseWriter()
	handler.ServeHTTP(writer, req)

	switch writer.respCode {
	case http.StatusOK:
		openApiSpec := &spec.Swagger{}
		if err := json.Unmarshal(writer.data, openApiSpec); err != nil {
			return nil, err
		}
		return openApiSpec, nil
	default:
		return nil, fmt.Errorf("failed to retrive openAPI spec, http error: %s", writer.String())
	}
}

// loadApiServiceSpec loads OpenAPI spec for the given API Service and then updates aggregator's spec.
func (s *openAPIAggregator) loadApiServiceSpec(handler http.Handler, apiService *apiregistration.APIService) error {

	// Ignore local services
	if apiService.Spec.Service == nil {
		return nil
	}

	openApiSpec, err := s.downloadOpenAPISpec(handler)
	if err != nil {
		return err
	}
	aggregator.FilterSpecByPaths(openApiSpec, []string{"/apis/" + apiService.Spec.Group + "/"})

	s.openAPISpecs[apiService.Name] = &openAPISpecInfo{
		apiService: *apiService,
		spec:       openApiSpec,
	}

	err = s.updateOpenAPISpec()
	if err != nil {
		delete(s.openAPISpecs, apiService.Name)
		return err
	}
	return nil
}
