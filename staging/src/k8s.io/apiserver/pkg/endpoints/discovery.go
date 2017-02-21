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

package endpoints

import (
	"bytes"
	"fmt"
	"io"
	"net/http"

	"github.com/emicklei/go-restful"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/endpoints/handlers"
	"k8s.io/apiserver/pkg/endpoints/handlers/negotiation"
	"k8s.io/apiserver/pkg/endpoints/handlers/responsewriters"
)

// AddApiWebService adds a service to return the supported api versions at the legacy /api.
func AddApiWebService(s runtime.NegotiatedSerializer, container *restful.Container, apiPrefix string, getAPIVersionsFunc func(req *restful.Request) *metav1.APIVersions) {
	// TODO: InstallREST should register each version automatically

	// Because in release 1.1, /api returns response with empty APIVersion, we
	// use StripVersionNegotiatedSerializer to keep the response backwards
	// compatible.
	mediaTypes, _ := negotiation.MediaTypesForSerializer(s)
	ss := stripVersionNegotiatedSerializer{s}
	versionHandler := APIVersionHandler(ss, getAPIVersionsFunc)
	ws := new(restful.WebService)
	ws.Path(apiPrefix)
	ws.Doc("get available API versions")
	ws.Route(ws.GET("/").To(versionHandler).
		Doc("get available API versions").
		Operation("getAPIVersions").
		Produces(mediaTypes...).
		Consumes(mediaTypes...).
		Writes(metav1.APIVersions{}))
	container.Add(ws)
}

// stripVersionEncoder strips APIVersion field from the encoding output. It's
// used to keep the responses at the discovery endpoints backward compatible
// with release-1.1, when the responses have empty APIVersion.
type stripVersionEncoder struct {
	encoder    runtime.Encoder
	serializer runtime.Serializer
}

func (c stripVersionEncoder) Encode(obj runtime.Object, w io.Writer) error {
	buf := bytes.NewBuffer([]byte{})
	err := c.encoder.Encode(obj, buf)
	if err != nil {
		return err
	}
	roundTrippedObj, gvk, err := c.serializer.Decode(buf.Bytes(), nil, nil)
	if err != nil {
		return err
	}
	gvk.Group = ""
	gvk.Version = ""
	roundTrippedObj.GetObjectKind().SetGroupVersionKind(*gvk)
	return c.serializer.Encode(roundTrippedObj, w)
}

// stripVersionNegotiatedSerializer will return stripVersionEncoder when
// EncoderForVersion is called. See comments for stripVersionEncoder.
type stripVersionNegotiatedSerializer struct {
	runtime.NegotiatedSerializer
}

func (n stripVersionNegotiatedSerializer) EncoderForVersion(encoder runtime.Encoder, gv runtime.GroupVersioner) runtime.Encoder {
	serializer, ok := encoder.(runtime.Serializer)
	if !ok {
		// The stripVersionEncoder needs both an encoder and decoder, but is called from a context that doesn't have access to the
		// decoder. We do a best effort cast here (since this code path is only for backwards compatibility) to get access to the caller's
		// decoder.
		panic(fmt.Sprintf("Unable to extract serializer from %#v", encoder))
	}
	versioned := n.NegotiatedSerializer.EncoderForVersion(encoder, gv)
	return stripVersionEncoder{versioned, serializer}
}

func keepUnversioned(group string) bool {
	return group == "" || group == "extensions"
}

// NewApisWebService returns a webservice serving the available api version under /apis.
func NewApisWebService(s runtime.NegotiatedSerializer, apiPrefix string, f func(req *restful.Request) []metav1.APIGroup) *restful.WebService {
	// Because in release 1.1, /apis returns response with empty APIVersion, we
	// use StripVersionNegotiatedSerializer to keep the response backwards
	// compatible.
	ss := stripVersionNegotiatedSerializer{s}
	mediaTypes, _ := negotiation.MediaTypesForSerializer(s)
	rootAPIHandler := handlers.RootAPIHandler(ss, f)
	ws := new(restful.WebService)
	ws.Path(apiPrefix)
	ws.Doc("get available API versions")
	ws.Route(ws.GET("/").To(rootAPIHandler).
		Doc("get available API versions").
		Operation("getAPIVersions").
		Produces(mediaTypes...).
		Consumes(mediaTypes...).
		Writes(metav1.APIGroupList{}))
	return ws
}

// NewGroupWebService returns a webservice serving the supported versions, preferred version, and name
// of a group. E.g., such a web service will be registered at /apis/extensions.
func NewGroupWebService(s runtime.NegotiatedSerializer, path string, group metav1.APIGroup) *restful.WebService {
	ss := s
	if keepUnversioned(group.Name) {
		// Because in release 1.1, /apis/extensions returns response with empty
		// APIVersion, we use StripVersionNegotiatedSerializer to keep the
		// response backwards compatible.
		ss = stripVersionNegotiatedSerializer{s}
	}
	mediaTypes, _ := negotiation.MediaTypesForSerializer(s)
	groupHandler := handlers.GroupHandler(ss, group)
	ws := new(restful.WebService)
	ws.Path(path)
	ws.Doc("get information of a group")
	ws.Route(ws.GET("/").To(groupHandler).
		Doc("get information of a group").
		Operation("getAPIGroup").
		Produces(mediaTypes...).
		Consumes(mediaTypes...).
		Writes(metav1.APIGroup{}))
	return ws
}

// Adds a service to return the supported resources, E.g., a such web service
// will be registered at /apis/extensions/v1.
func AddSupportedResourcesWebService(s runtime.NegotiatedSerializer, ws *restful.WebService, groupVersion schema.GroupVersion, lister handlers.APIResourceLister) {
	ss := s
	if keepUnversioned(groupVersion.Group) {
		// Because in release 1.1, /apis/extensions/v1beta1 returns response
		// with empty APIVersion, we use StripVersionNegotiatedSerializer to
		// keep the response backwards compatible.
		ss = stripVersionNegotiatedSerializer{s}
	}
	mediaTypes, _ := negotiation.MediaTypesForSerializer(s)
	resourceHandler := handlers.SupportedResourcesHandler(ss, groupVersion, lister)
	ws.Route(ws.GET("/").To(resourceHandler).
		Doc("get available resources").
		Operation("getAPIResources").
		Produces(mediaTypes...).
		Consumes(mediaTypes...).
		Writes(metav1.APIResourceList{}))
}

// APIVersionHandler returns a handler which will list the provided versions as available.
func APIVersionHandler(s runtime.NegotiatedSerializer, getAPIVersionsFunc func(req *restful.Request) *metav1.APIVersions) restful.RouteFunction {
	return func(req *restful.Request, resp *restful.Response) {
		responsewriters.WriteObjectNegotiated(s, schema.GroupVersion{}, resp.ResponseWriter, req.Request, http.StatusOK, getAPIVersionsFunc(req))
	}
}
