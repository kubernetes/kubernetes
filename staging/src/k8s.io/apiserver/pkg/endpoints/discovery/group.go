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

	"github.com/emicklei/go-restful"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/endpoints/handlers/negotiation"
	"k8s.io/apiserver/pkg/endpoints/handlers/responsewriters"
)

// apiGroupHandler creates a webservice serving the supported versions, preferred version, and name
// of a group. E.g., such a web service will be registered at /apis/extensions.
type apiGroupHandler struct {
	serializer runtime.NegotiatedSerializer

	group metav1.APIGroup
}

func NewAPIGroupHandler(serializer runtime.NegotiatedSerializer, group metav1.APIGroup) *apiGroupHandler {
	if keepUnversioned(group.Name) {
		// Because in release 1.1, /apis/extensions returns response with empty
		// APIVersion, we use stripVersionNegotiatedSerializer to keep the
		// response backwards compatible.
		serializer = stripVersionNegotiatedSerializer{serializer}
	}

	return &apiGroupHandler{
		serializer: serializer,
		group:      group,
	}
}

func (s *apiGroupHandler) WebService() *restful.WebService {
	mediaTypes, _ := negotiation.MediaTypesForSerializer(s.serializer)
	ws := new(restful.WebService)
	ws.Path(APIGroupPrefix + "/" + s.group.Name)
	ws.Doc("get information of a group")
	ws.Route(ws.GET("/").To(s.handle).
		Doc("get information of a group").
		Operation("getAPIGroup").
		Produces(mediaTypes...).
		Consumes(mediaTypes...).
		Writes(metav1.APIGroup{}))
	return ws
}

// handle returns a handler which will return the api.GroupAndVersion of the group.
func (s *apiGroupHandler) handle(req *restful.Request, resp *restful.Response) {
	responsewriters.WriteObjectNegotiated(s.serializer, schema.GroupVersion{}, resp.ResponseWriter, req.Request, http.StatusOK, &s.group)
}
