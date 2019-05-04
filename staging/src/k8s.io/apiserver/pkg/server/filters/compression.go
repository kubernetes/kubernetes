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

package filters

import (
	"net/http"

	restful "github.com/emicklei/go-restful"

	"k8s.io/apiserver/pkg/endpoints/request"
	handlers "k8s.io/apiserver/third_party/gorilla-handlers"
)

// WithCompression wraps an http.Handler with the Compression Handler
func WithCompression(handler http.Handler) http.Handler {
	withCompression := handlers.CompressHandler(handler)
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		ctx := req.Context()
		if info, ok := request.RequestInfoFrom(ctx); !ok || !info.IsResourceRequest || info.Verb == "watch" {
			handler.ServeHTTP(w, req)
			return
		}
		withCompression.ServeHTTP(w, req)
	})
}

// RestfulWithCompression wraps WithCompression to be compatible with go-restful
func RestfulWithCompression(function restful.RouteFunction) restful.RouteFunction {
	return restful.RouteFunction(func(request *restful.Request, response *restful.Response) {
		handler := handlers.CompressHandler(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
			response.ResponseWriter = w
			request.Request = req
			function(request, response)
		}))
		handler.ServeHTTP(response.ResponseWriter, request.Request)
	})
}
