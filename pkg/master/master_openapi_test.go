// +build !race

/*
Copyright 2014 The Kubernetes Authors.

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

package master

// This test file is separated from master_test.go so we would be able to disable
// race check for it. TestValidOpenAPISpec will became extremely slow if -race
// flag exists, and will cause the tests to timeout.

import (
	"net/http"
	"net/http/httptest"
	"testing"

	apirequest "k8s.io/apiserver/pkg/endpoints/request"
	genericapiserver "k8s.io/apiserver/pkg/server"
	"k8s.io/kubernetes/pkg/api"
	openapigen "k8s.io/kubernetes/pkg/generated/openapi"

	"github.com/go-openapi/loads"
	"github.com/go-openapi/spec"
	"github.com/go-openapi/strfmt"
	"github.com/go-openapi/validate"
)

// TestValidOpenAPISpec verifies that the open api is added
// at the proper endpoint and the spec is valid.
func TestValidOpenAPISpec(t *testing.T) {
	etcdserver, config, sharedInformers, assert := setUp(t)
	defer etcdserver.Terminate(t)

	config.GenericConfig.EnableIndex = true
	config.GenericConfig.OpenAPIConfig = genericapiserver.DefaultOpenAPIConfig(openapigen.GetOpenAPIDefinitions, api.Scheme)
	config.GenericConfig.OpenAPIConfig.Info = &spec.Info{
		InfoProps: spec.InfoProps{
			Title:   "Kubernetes",
			Version: "unversioned",
		},
	}
	config.GenericConfig.SwaggerConfig = genericapiserver.DefaultSwaggerConfig()

	master, err := config.Complete(sharedInformers).New(genericapiserver.EmptyDelegate)
	if err != nil {
		t.Fatalf("Error in bringing up the master: %v", err)
	}

	// make sure swagger.json is not registered before calling PrepareRun.
	server := httptest.NewServer(apirequest.WithRequestContext(master.GenericAPIServer.Handler.Director, master.GenericAPIServer.RequestContextMapper()))
	defer server.Close()
	resp, err := http.Get(server.URL + "/swagger.json")
	if !assert.NoError(err) {
		t.Errorf("unexpected error: %v", err)
	}
	assert.Equal(http.StatusNotFound, resp.StatusCode)

	master.GenericAPIServer.PrepareRun()

	resp, err = http.Get(server.URL + "/swagger.json")
	if !assert.NoError(err) {
		t.Errorf("unexpected error: %v", err)
	}
	assert.Equal(http.StatusOK, resp.StatusCode)

	// as json schema
	var sch spec.Schema
	if assert.NoError(decodeResponse(resp, &sch)) {
		validator := validate.NewSchemaValidator(spec.MustLoadSwagger20Schema(), nil, "", strfmt.Default)
		res := validator.Validate(&sch)
		assert.NoError(res.AsError())
	}

	// Validate OpenApi spec
	doc, err := loads.Spec(server.URL + "/swagger.json")
	if assert.NoError(err) {
		validator := validate.NewSpecValidator(doc.Schema(), strfmt.Default)
		res, warns := validator.Validate(doc)
		assert.NoError(res.AsError())
		if !warns.IsValid() {
			t.Logf("Open API spec on root has some warnings : %v", warns)
		}
	}
}
