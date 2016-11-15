/*
Copyright 2016 The Kubernetes Authors.

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

package mux

import (
	"net/http"

	"github.com/emicklei/go-restful"

	"k8s.io/kubernetes/pkg/apiserver"
	"k8s.io/kubernetes/pkg/runtime"
)

// APIContainer is a restful container which in addition support registering
// handlers that do not show up in swagger or in /
type APIContainer struct {
	*restful.Container
	NonSwaggerRoutes PathRecorderMux
	SecretRoutes     Mux
}

// NewAPIContainer constructs a new container for APIs
func NewAPIContainer(mux *http.ServeMux, s runtime.NegotiatedSerializer) *APIContainer {
	c := APIContainer{
		Container: restful.NewContainer(),
		NonSwaggerRoutes: PathRecorderMux{
			mux: mux,
		},
		SecretRoutes: mux,
	}
	c.Container.ServeMux = mux
	c.Container.Router(restful.CurlyRouter{}) // e.g. for proxy/{kind}/{name}/{*}

	apiserver.InstallRecoverHandler(s, c.Container)
	apiserver.InstallServiceErrorHandler(s, c.Container)

	return &c
}
