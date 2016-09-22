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
	"github.com/emicklei/go-restful"

	"k8s.io/kubernetes/pkg/genericapiserver/routes/data/swagger"
)

// SwaggerUI exposes files in third_party/swagger-ui/ under /swagger-ui.
func SwaggerUI() *restful.WebService {
	prefix := "/swagger-ui"
	handler := http.StripPrefix(prefix, http.FileServer(&assetfs.AssetFS{
		Asset:    swagger.Asset,
		AssetDir: swagger.AssetDir,
		Prefix:   "third_party/swagger-ui",
	}))

	ws := new(restful.WebService)
	ws.Path(prefix)
	ws.Doc("swagger user interface")
	wildcard := "/{subpath:*}" // go-restful curly path
	ws.Route(ws.GET(wildcard).To(HandlerRouteFunction(handler.ServeHTTP)))
	ws.Route(ws.HEAD(wildcard).To(HandlerRouteFunction(handler.ServeHTTP))) // used for eTags
	ws.Route(ws.GET("/").To(HandlerRouteFunction(handler.ServeHTTP)))
	ws.Route(ws.HEAD("/").To(HandlerRouteFunction(handler.ServeHTTP))) // used for eTags
	return ws
}
