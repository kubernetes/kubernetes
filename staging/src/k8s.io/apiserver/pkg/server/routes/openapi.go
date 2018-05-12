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
	restful "github.com/emicklei/go-restful"
	"github.com/golang/glog"

	"k8s.io/apiserver/pkg/server/mux"
	"k8s.io/kube-openapi/pkg/common"
	"k8s.io/kube-openapi/pkg/handler"
)

// OpenAPI installs spec endpoints for each web service.
type OpenAPI struct {
	Config *common.Config
}

// Install adds the SwaggerUI webservice to the given mux.
func (oa OpenAPI) Install(c *restful.Container, mux *mux.PathRecorderMux) {
	// NOTE: [DEPRECATION] We will announce deprecation for format-separated endpoints for OpenAPI spec,
	// and switch to a single /openapi/v2 endpoint in Kubernetes 1.10. The design doc and deprecation process
	// are tracked at: https://docs.google.com/document/d/19lEqE9lc4yHJ3WJAJxS_G7TcORIJXGHyq3wpwcH28nU.
	_, err := handler.BuildAndRegisterOpenAPIService("/swagger.json", c.RegisteredWebServices(), oa.Config, mux)
	if err != nil {
		glog.Fatalf("Failed to register open api spec for root: %v", err)
	}
	_, err = handler.BuildAndRegisterOpenAPIVersionedService("/openapi/v2", c.RegisteredWebServices(), oa.Config, mux)
	if err != nil {
		glog.Fatalf("Failed to register versioned open api spec for root: %v", err)
	}
}
