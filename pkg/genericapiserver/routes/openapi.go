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
	"strings"

	"k8s.io/kubernetes/pkg/genericapiserver/mux"
	"k8s.io/kubernetes/pkg/genericapiserver/openapi"
	"k8s.io/kubernetes/pkg/genericapiserver/openapi/common"

	"github.com/emicklei/go-restful"
	"github.com/go-openapi/spec"
	"github.com/golang/glog"
)

// OpenAPI installs spec endpoints for each web service.
type OpenAPI struct {
	Config *common.Config
}

// Install adds the SwaggerUI webservice to the given mux.
func (oa OpenAPI) Install(c *mux.APIContainer) {
	// Install one spec per web service, an ideal client will have a ClientSet containing one client
	// per each of these specs.
	for _, w := range c.RegisteredWebServices() {
		if strings.HasPrefix(w.RootPath(), "/swaggerapi") {
			continue
		}

		config := *oa.Config
		config.Info = new(spec.Info)
		*config.Info = *oa.Config.Info
		config.Info.Title = config.Info.Title + " " + w.RootPath()
		err := openapi.RegisterOpenAPIService(w.RootPath()+"/swagger.json", []*restful.WebService{w}, &config, c)
		if err != nil {
			glog.Fatalf("Failed to register open api spec for %v: %v", w.RootPath(), err)
		}
	}
	err := openapi.RegisterOpenAPIService("/swagger.json", c.RegisteredWebServices(), oa.Config, c)
	if err != nil {
		glog.Fatalf("Failed to register open api spec for root: %v", err)
	}
}
