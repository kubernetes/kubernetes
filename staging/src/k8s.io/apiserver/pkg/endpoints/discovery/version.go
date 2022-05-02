/*
Copyright 2017 The Kubernetes Authors.

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

package discovery

import (
	"net/http"

	restful "github.com/emicklei/go-restful"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/endpoints/handlers/negotiation"
	"k8s.io/klog/v2"
)

// APIVersionHandler creates a webservice serving the supported resources for the version
// E.g., such a web service will be registered at /apis/extensions/v1beta1.
type APIVersionHandler struct {
	serializer runtime.NegotiatedSerializer

	groupVersion schema.GroupVersion
	resourceList *metav1.APIResourceList
	hash         string
}

func NewAPIVersionHandler(
	serializer runtime.NegotiatedSerializer,
	groupVersion schema.GroupVersion,
	resources []metav1.APIResource,
) *APIVersionHandler {

	resourceList := &metav1.APIResourceList{
		GroupVersion: groupVersion.String(),
		APIResources: resources,
	}

	if keepUnversioned(groupVersion.Group) {
		// Because in release 1.1, /apis/extensions returns response with empty
		// APIVersion, we use stripVersionNegotiatedSerializer to keep the
		// response backwards compatible.
		serializer = stripVersionNegotiatedSerializer{serializer}
	}

	hash, err := CalculateETag(resourceList)
	if err != nil {
		// This method cannot fail. If E-Tag cannot be calculated, then we will
		// simply not support etags with this endpoint.
		klog.Error(
			"failed to calculate e-tag for resoure list of %v. E-Tags will not be supported",
			groupVersion.String())

		hash = ""
	}

	return &APIVersionHandler{
		serializer:   serializer,
		groupVersion: groupVersion,
		resourceList: resourceList,
		hash:         hash,
	}
}

func (s *APIVersionHandler) AddToWebService(ws *restful.WebService) {
	mediaTypes, _ := negotiation.MediaTypesForSerializer(s.serializer)
	ws.Route(ws.GET("/").To(s.handle).
		Doc("get available resources").
		Operation("getAPIResources").
		Produces(mediaTypes...).
		Consumes(mediaTypes...).
		Writes(metav1.APIResourceList{}))
}

// handle returns a handler which will return the api.VersionAndVersion of the group.
func (s *APIVersionHandler) handle(req *restful.Request, resp *restful.Response) {
	s.ServeHTTP(resp.ResponseWriter, req.Request)
}

func (s *APIVersionHandler) GetCurrentHash() string {
	return s.hash
}

func (s *APIVersionHandler) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	ServeHTTPWithETag(s.resourceList, s.hash, s.serializer, w, req)
}
