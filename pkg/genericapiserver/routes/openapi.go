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
	"k8s.io/kubernetes/pkg/apiserver"
	"k8s.io/kubernetes/pkg/genericapiserver/openapi"

	"github.com/emicklei/go-restful"
	"github.com/go-openapi/spec"
	"github.com/golang/glog"
)

type OpenAPI struct {
	Info            spec.Info
	DefaultResponse spec.Response
}

func (oa OpenAPI) Install(mux *apiserver.PathRecorderMux, c *restful.Container) {
	// Install one spec per web service, an ideal client will have a ClientSet containing one client
	// per each of these specs.
	for _, w := range c.RegisteredWebServices() {
		if w.RootPath() == "/swaggerapi" {
			continue
		}
		info := oa.Info
		info.Title = info.Title + " " + w.RootPath()
		err := openapi.RegisterOpenAPIService(&openapi.Config{
			OpenAPIServePath: w.RootPath() + "/swagger.json",
			WebServices:      []*restful.WebService{w},
			ProtocolList:     []string{"https"},
			IgnorePrefixes:   []string{"/swaggerapi"},
			Info:             &info,
			DefaultResponse:  &oa.DefaultResponse,
		}, c)
		if err != nil {
			glog.Fatalf("Failed to register open api spec for %v: %v", w.RootPath(), err)
		}
	}
	err := openapi.RegisterOpenAPIService(&openapi.Config{
		OpenAPIServePath: "/swagger.json",
		WebServices:      c.RegisteredWebServices(),
		ProtocolList:     []string{"https"},
		IgnorePrefixes:   []string{"/swaggerapi"},
		Info:             &oa.Info,
		DefaultResponse:  &oa.DefaultResponse,
	}, c)
	if err != nil {
		glog.Fatalf("Failed to register open api spec for root: %v", err)
	}
}
