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

package handlers

import (
	"net/http"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/endpoints/handlers/responsewriters"

	"github.com/emicklei/go-restful"
)

type APIResourceLister interface {
	ListAPIResources() []metav1.APIResource
}

// RootAPIHandler returns a handler which will list the provided groups and versions as available.
func RootAPIHandler(s runtime.NegotiatedSerializer, f func(req *restful.Request) []metav1.APIGroup) restful.RouteFunction {
	return func(req *restful.Request, resp *restful.Response) {
		responsewriters.WriteObjectNegotiated(s, schema.GroupVersion{}, resp.ResponseWriter, req.Request, http.StatusOK, &metav1.APIGroupList{Groups: f(req)})
	}
}

// GroupHandler returns a handler which will return the api.GroupAndVersion of
// the group.
func GroupHandler(s runtime.NegotiatedSerializer, group metav1.APIGroup) restful.RouteFunction {
	return func(req *restful.Request, resp *restful.Response) {
		responsewriters.WriteObjectNegotiated(s, schema.GroupVersion{}, resp.ResponseWriter, req.Request, http.StatusOK, &group)
	}
}

// SupportedResourcesHandler returns a handler which will list the provided resources as available.
func SupportedResourcesHandler(s runtime.NegotiatedSerializer, groupVersion schema.GroupVersion, lister APIResourceLister) restful.RouteFunction {
	return func(req *restful.Request, resp *restful.Response) {
		responsewriters.WriteObjectNegotiated(s, schema.GroupVersion{}, resp.ResponseWriter, req.Request, http.StatusOK, &metav1.APIResourceList{GroupVersion: groupVersion.String(), APIResources: lister.ListAPIResources()})
	}
}
