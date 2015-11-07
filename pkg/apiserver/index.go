/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package apiserver

import (
	"net/http"
	"sort"

	"k8s.io/kubernetes/pkg/api/unversioned"

	"github.com/emicklei/go-restful"
)

func IndexHandler(container *restful.Container, muxHelper *MuxHelper) func(http.ResponseWriter, *http.Request) {
	return func(w http.ResponseWriter, r *http.Request) {
		status := http.StatusOK
		if r.URL.Path != "/" && r.URL.Path != "/index.html" {
			// Since "/" matches all paths, handleIndex is called for all paths for which there is no handler registered.
			// We want to to return a 404 status with a list of all valid paths, incase of an invalid URL request.
			status = http.StatusNotFound
		}
		var handledPaths []string
		// Extract the paths handled using restful.WebService
		for _, ws := range container.RegisteredWebServices() {
			handledPaths = append(handledPaths, ws.RootPath())
		}
		// Extract the paths handled using mux handler.
		handledPaths = append(handledPaths, muxHelper.RegisteredPaths...)
		sort.Strings(handledPaths)
		writeRawJSON(status, unversioned.RootPaths{Paths: handledPaths}, w)
	}
}
