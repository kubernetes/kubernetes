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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/endpoints/handlers/responsewriters"
	"k8s.io/apiserver/pkg/server/mux"
)

// ListedPathProvider is an interface for providing paths that should be reported at /.
type ListedPathProvider interface {
	// ListedPaths is an alphabetically sorted list of paths to be reported at /.
	ListedPaths() []string
}

// ListedPathProviders is a convenient way to combine multiple ListedPathProviders
type ListedPathProviders []ListedPathProvider

// ListedPaths unions and sorts the included paths.
func (p ListedPathProviders) ListedPaths() []string {
	ret := sets.String{}
	for _, provider := range p {
		for _, path := range provider.ListedPaths() {
			ret.Insert(path)
		}
	}

	return ret.List()
}

// Index provides a webservice for the http root / listing all known paths.
type Index struct{}

// Install adds the Index webservice to the given mux.
func (i Index) Install(pathProvider ListedPathProvider, mux *mux.PathRecorderMux, delegate http.Handler) {
	mux.UnlistedHandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		status := http.StatusOK
		if r.URL.Path != "/" && r.URL.Path != "/index.html" {
			// Since "/" matches all paths, handleIndex is called for all paths for which there is no handler api.Registry.
			// if we have a delegate, we should call to it and simply return
			if delegate != nil {
				delegate.ServeHTTP(w, r)
				return
			}

			// If we have no delegate, we want to return a 404 status with a list of all valid paths, incase of an invalid URL request.
			status = http.StatusNotFound
		}
		responsewriters.WriteRawJSON(status, metav1.RootPaths{Paths: pathProvider.ListedPaths()}, w)
	})
}
