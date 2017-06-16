/*
Copyright 2016 The Kubernetes Authors.

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

package routes

import (
	"github.com/emicklei/go-restful"

	"k8s.io/apimachinery/pkg/openapi"
	"k8s.io/apiserver/pkg/server/mux"
	apiserveropenapi "k8s.io/apiserver/pkg/server/openapi"

	"github.com/golang/glog"
)

// OpenAPI installs spec endpoints for each web service.
type OpenAPI struct {
	Config *openapi.Config
}

// Install adds the SwaggerUI webservice to the given mux.
func (oa OpenAPI) Install(c *restful.Container, mux *mux.PathRecorderMux) *apiserveropenapi.OpenAPIService {
	openapiSpec, err := apiserveropenapi.BuildSwaggerSpec(c.RegisteredWebServices(), oa.Config)
	if err != nil {
		glog.Fatalf("Failed to register open api spec for root: %v", err)
		return nil
	}
	service, err := apiserveropenapi.RegisterOpenAPIService(openapiSpec, "/swagger.json", mux)
	if err != nil {
		glog.Fatalf("Failed to register open api spec for root: %v", err)
		return nil
	}
	return service
}
