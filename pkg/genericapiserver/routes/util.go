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
	"strings"

	"net/http"

	"github.com/emicklei/go-restful"
)

// HandlerRouteFunction wraps a net/http HandlerFunc into a restful RouteFunction
func HandlerRouteFunction(h http.HandlerFunc) restful.RouteFunction {
	return func(req *restful.Request, resp *restful.Response) {
		h(resp.ResponseWriter, req.Request)
	}
}

// WebServiceGETMux support adding a GET handler to a restful WebService. The
// path is stripped from registered handler paths.
type WebServiceGETMux struct {
	WS   *restful.WebService
	Path string
}

// Handle registers the given handler under the given pattern as GET handler.
func (ws WebServiceGETMux) Handle(pattern string, handler http.Handler) {
	if len(ws.Path) > 0 && strings.HasPrefix(pattern, ws.Path) {
		pattern = pattern[len(ws.Path):]
		if pattern == "" {
			pattern = "/"
		}
	}
	ws.WS.Route(ws.WS.GET(pattern).To(HandlerRouteFunction(handler.ServeHTTP)))
}
