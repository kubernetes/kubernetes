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

package routes

import (
	"net/http"

	assetfs "github.com/elazarl/go-bindata-assetfs"

	"k8s.io/apiserver/pkg/server/mux"
	"k8s.io/apiserver/pkg/server/routes/data/swagger"
)

// SwaggerUI exposes files in third_party/swagger-ui/ under /swagger-ui.
type SwaggerUI struct{}

// Install adds the SwaggerUI webservice to the given mux.
func (l SwaggerUI) Install(c *mux.PathRecorderMux) {
	fileServer := http.FileServer(&assetfs.AssetFS{
		Asset:    swagger.Asset,
		AssetDir: swagger.AssetDir,
		Prefix:   "third_party/swagger-ui",
	})
	prefix := "/swagger-ui/"
	c.Handle(prefix, http.StripPrefix(prefix, fileServer))
}
