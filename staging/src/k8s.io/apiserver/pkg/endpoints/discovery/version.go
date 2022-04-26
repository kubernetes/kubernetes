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
	"strconv"
	"time"

	restful "github.com/emicklei/go-restful"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/endpoints/handlers/negotiation"
	"k8s.io/apiserver/pkg/endpoints/handlers/responsewriters"
)

type APIResourceLister interface {
	ListAPIResources() ([]metav1.APIResource, string)
}

type APIResourceListerFunc func() ([]metav1.APIResource, string)

func (f APIResourceListerFunc) ListAPIResources() ([]metav1.APIResource, string) {
	return f()
}

// APIVersionHandler creates a webservice serving the supported resources for the version
// E.g., such a web service will be registered at /apis/extensions/v1beta1.
type APIVersionHandler struct {
	serializer runtime.NegotiatedSerializer

	groupVersion      schema.GroupVersion
	apiResourceLister APIResourceLister
}

func NewAPIVersionHandler(serializer runtime.NegotiatedSerializer, groupVersion schema.GroupVersion, apiResourceLister APIResourceLister) *APIVersionHandler {
	if keepUnversioned(groupVersion.Group) {
		// Because in release 1.1, /apis/extensions returns response with empty
		// APIVersion, we use stripVersionNegotiatedSerializer to keep the
		// response backwards compatible.
		serializer = stripVersionNegotiatedSerializer{serializer}
	}

	return &APIVersionHandler{
		serializer:        serializer,
		groupVersion:      groupVersion,
		apiResourceLister: apiResourceLister,
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
	_, hash := s.apiResourceLister.ListAPIResources()
	return hash
}

func (s *APIVersionHandler) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	reqURL := req.URL
	if reqURL == nil {
		// Can not find documentation guaranteeing the non-nility of reqURL.
		w.WriteHeader(500)
		return
	}

	// Get current resource list and hash
	resourceList, hash := s.apiResourceLister.ListAPIResources()

	if providedHash := reqURL.Query().Get("hash"); len(providedHash) > 0 && len(hash) > 0 {
		if hash == providedHash {
			// The Vary header is required because the Accept header can
			// change the contents returned. This prevents clients from caching
			// protobuf as JSON and vice versa.
			w.Header().Set("Vary", "Accept")

			// Only set these headers when a hash is given.
			w.Header().Set("Cache-Control", "public, immutable")

			// Set the Expires directive to the maximum value of one year from the request,
			// effectively indicating that the cache never expires.
			w.Header().Set(
				"Expires", time.Now().AddDate(1, 0, 0).Format(time.RFC1123))
		} else {
			// redirect with reply
			redirectURL := *reqURL
			query := redirectURL.Query()
			query.Set("hash", hash)
			redirectURL.RawQuery = query.Encode()

			//!TODO: is MovedPermanently the right status code to use here?
			http.Redirect(w, req, redirectURL.String(), http.StatusMovedPermanently)
			return
		}
	}

	// ETag must be enclosed in double quotes:
	// https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/ETag
	w.Header().Set("Etag", strconv.Quote(hash))

	responsewriters.WriteObjectNegotiated(
		s.serializer,
		negotiation.DefaultEndpointRestrictions,
		schema.GroupVersion{},
		w,
		req,
		http.StatusOK,
		&metav1.APIResourceList{
			GroupVersion: s.groupVersion.String(),
			APIResources: resourceList,
		},
	)
}
