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
func (i Index) Install(pathProvider ListedPathProvider, mux *mux.PathRecorderMux) {
	handler := IndexLister{StatusCode: http.StatusOK, PathProvider: pathProvider}

	mux.UnlistedHandle("/", handler)
	mux.UnlistedHandle("/index.html", handler)
}

// IndexLister lists the available indexes with the status code provided
type IndexLister struct {
	StatusCode   int
	PathProvider ListedPathProvider
}

// ServeHTTP serves the available paths.
func (i IndexLister) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	responsewriters.WriteRawJSON(i.StatusCode, metav1.RootPaths{Paths: i.PathProvider.ListedPaths()}, w)
}
