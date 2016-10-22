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
	"net/http"
	"testing"

	"github.com/stretchr/testify/assert"

	genericmux "k8s.io/kubernetes/pkg/genericapiserver/mux"
)

// TestInstallSwaggerAPI verifies that the swagger api is added
// at the proper endpoint.
func TestInstallSwaggerAPI(t *testing.T) {
	assert := assert.New(t)

	// Install swagger and test
	c := genericmux.NewAPIContainer(http.NewServeMux(), nil)
	Swagger{ExternalAddress: "1.2.3.4"}.Install(c)
	ws := c.RegisteredWebServices()
	if assert.NotEqual(0, len(ws), "SwaggerAPI not installed.") {
		assert.Equal("/swaggerapi/", ws[0].RootPath(), "SwaggerAPI did not install to the proper path. %s != /swaggerapi", ws[0].RootPath())
	}

	// Empty externalHost verification
	c = genericmux.NewAPIContainer(http.NewServeMux(), nil)
	Swagger{ExternalAddress: ""}.Install(c)
	ws = c.RegisteredWebServices()
	if assert.NotEqual(0, len(ws), "SwaggerAPI not installed.") {
		assert.Equal("/swaggerapi/", ws[0].RootPath(), "SwaggerAPI did not install to the proper path. %s != /swaggerapi", ws[0].RootPath())
	}
}
