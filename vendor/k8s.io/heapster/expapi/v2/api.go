// Copyright 2015 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package v2

import (
	"k8s.io/heapster/manager"

	restful "github.com/emicklei/go-restful"
	"github.com/golang/glog"
)

type Api struct {
	manager manager.Manager
}

// Create a new Api to serve from the specified cache.
func NewApi(m manager.Manager) *Api {
	return &Api{
		manager: m,
	}
}

// Register the Api on the specified endpoint.
func (a *Api) Register(container *restful.Container) {
	// Register the endpoints of the metrics
	a.RegisterMetrics(container)
}

func compressionFilter(req *restful.Request, resp *restful.Response, chain *restful.FilterChain) {
	// wrap responseWriter into a compressing one
	compress, err := restful.NewCompressingResponseWriter(resp.ResponseWriter, restful.ENCODING_GZIP)
	if err != nil {
		glog.Warningf("Failed to create CompressingResponseWriter for request %q: %v", req.Request.URL, err)
		return
	}
	resp.ResponseWriter = compress
	defer compress.Close()
	chain.ProcessFilter(req, resp)
}
