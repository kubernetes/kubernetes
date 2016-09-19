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
	"github.com/emicklei/go-restful"

	"net/http/pprof"
)

// Profiling adds handlers for pprof under /debug/pprof.
type Profiling struct{}

func (d Profiling) Install(c *restful.Container) {
	ws := new(restful.WebService)
	ws.Path("/debug/pprof/")
	ws.Doc("get go pprof debugging info")
	ws.Route(ws.GET("/").To(HandlerRouteFunction(pprof.Index)))
	ws.Route(ws.GET("/profile").To(HandlerRouteFunction(pprof.Profile)))
	ws.Route(ws.GET("/symbol").To(HandlerRouteFunction(pprof.Symbol)))

	c.Add(ws)
}
