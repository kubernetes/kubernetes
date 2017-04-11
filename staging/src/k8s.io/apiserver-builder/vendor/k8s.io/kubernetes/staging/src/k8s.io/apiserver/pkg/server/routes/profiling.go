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
	"net/http/pprof"

	"k8s.io/apiserver/pkg/server/mux"
)

// Profiling adds handlers for pprof under /debug/pprof.
type Profiling struct{}

// Install adds the Profiling webservice to the given mux.
func (d Profiling) Install(c *mux.PathRecorderMux) {
	c.UnlistedHandleFunc("/debug/pprof/", pprof.Index)
	c.UnlistedHandleFunc("/debug/pprof/profile", pprof.Profile)
	c.UnlistedHandleFunc("/debug/pprof/symbol", pprof.Symbol)
	c.UnlistedHandleFunc("/debug/pprof/trace", pprof.Trace)
}
