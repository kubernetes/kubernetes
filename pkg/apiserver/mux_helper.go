/*
Copyright 2015 The Kubernetes Authors.

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
)

// Offers additional functionality over ServeMux, for ex: supports listing registered paths.
type MuxHelper struct {
	Mux             Mux
	RegisteredPaths []string
}

func (m *MuxHelper) Handle(path string, handler http.Handler) {
	m.RegisteredPaths = append(m.RegisteredPaths, path)
	m.Mux.Handle(path, handler)
}

func (m *MuxHelper) HandleFunc(path string, handler func(http.ResponseWriter, *http.Request)) {
	m.RegisteredPaths = append(m.RegisteredPaths, path)
	m.Mux.HandleFunc(path, handler)
}
