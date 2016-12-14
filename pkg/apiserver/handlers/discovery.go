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
	"strings"

	metav1 "k8s.io/kubernetes/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/apiserver/handlers/responsewriters"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/runtime/schema"
	"k8s.io/kubernetes/pkg/util/sets"
	utilversion "k8s.io/kubernetes/pkg/util/version"

	"github.com/emicklei/go-restful"
)

type APIResourceLister interface {
	ListAPIResources() []metav1.APIResource
}

// RootAPIHandler returns a handler which will list the provided groups and versions as available.
func RootAPIHandler(s runtime.NegotiatedSerializer, f func(req *restful.Request) []metav1.APIGroup) restful.RouteFunction {
	return func(req *restful.Request, resp *restful.Response) {
		responsewriters.WriteObjectNegotiated(s, schema.GroupVersion{}, resp.ResponseWriter, req.Request, http.StatusOK, &metav1.APIGroupList{Groups: filterAPIGroups(req, f(req))})
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

// TODO: Remove in 1.6. This is for backward compatibility with 1.4 kubectl.
// See https://github.com/kubernetes/kubernetes/issues/35791
var groupsWithNewVersionsIn1_5 = sets.NewString("apps", "policy")

// TODO: Remove in 1.6.
func filterAPIGroups(req *restful.Request, groups []metav1.APIGroup) []metav1.APIGroup {
	if !isOldKubectl(req.HeaderParameter("User-Agent")) {
		return groups
	}
	// hide API group that has new versions added in 1.5.
	var ret []metav1.APIGroup
	for _, group := range groups {
		if groupsWithNewVersionsIn1_5.Has(group.Name) {
			continue
		}
		ret = append(ret, group)
	}
	return ret
}

// TODO: Remove in 1.6. Returns if kubectl is older than v1.5.0
func isOldKubectl(userAgent string) bool {
	// example userAgent string: kubectl-1.3/v1.3.8 (linux/amd64) kubernetes/e328d5b
	if !strings.Contains(userAgent, "kubectl") {
		return false
	}
	userAgent = strings.Split(userAgent, " ")[0]
	subs := strings.Split(userAgent, "/")
	if len(subs) != 2 {
		return false
	}
	kubectlVersion, versionErr := utilversion.ParseSemantic(subs[1])
	if versionErr != nil {
		return false
	}
	return kubectlVersion.LessThan(utilversion.MustParseSemantic("v1.5.0"))
}
