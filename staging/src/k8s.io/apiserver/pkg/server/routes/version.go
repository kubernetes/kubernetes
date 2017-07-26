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

	"k8s.io/apimachinery/pkg/version"
	"k8s.io/apiserver/pkg/endpoints/handlers/responsewriters"
	"k8s.io/apiserver/pkg/server/mux"
)

// Version provides a webservice with version information.
type Version struct {
	Version *version.Info
}

// Install registers the APIServer's `/version` handler.
func (v Version) Install(c *mux.PathRecorderMux) {
	c.Handle("/version", v)
	c.UnlistedHandle("/version/", v)
}

// handleVersion writes the server's version information.
func (v Version) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	responsewriters.WriteRawJSON(http.StatusOK, *v.Version, w)
}
