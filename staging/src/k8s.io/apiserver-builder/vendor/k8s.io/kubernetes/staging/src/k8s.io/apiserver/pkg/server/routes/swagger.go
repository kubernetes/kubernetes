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
	"k8s.io/apiserver/pkg/server/mux"

	"github.com/emicklei/go-restful/swagger"
)

// Swagger installs the /swaggerapi/ endpoint to allow schema discovery
// and traversal. It is optional to allow consumers of the Kubernetes GenericAPIServer to
// register their own web services into the Kubernetes mux prior to initialization
// of swagger, so that other resource types show up in the documentation.
type Swagger struct {
	Config *swagger.Config
}

// Install adds the SwaggerUI webservice to the given mux.
func (s Swagger) Install(c *mux.APIContainer) {
	s.Config.WebServices = c.RegisteredWebServices()
	swagger.RegisterSwaggerService(*s.Config, c.Container)
}
