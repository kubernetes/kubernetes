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
	"bytes"
	"fmt"
	"net/http"
	rt "runtime"

	"github.com/emicklei/go-restful"
	"github.com/golang/glog"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/endpoints/handlers/responsewriters"
)

// APIContainer is a restful container which in addition support registering
// handlers that do not show up in swagger or in /
type APIContainer struct {
	*restful.Container

	// NonSwaggerRoutes are recorded and are visible at /, but do not show up in Swagger.
	NonSwaggerRoutes PathRecorderMux

	// UnlistedRoutes are not recorded, therefore not visible at / and do not show up in Swagger.
	UnlistedRoutes *http.ServeMux
}

// NewAPIContainer constructs a new container for APIs
func NewAPIContainer(mux *http.ServeMux, s runtime.NegotiatedSerializer) *APIContainer {
	c := APIContainer{
		Container: restful.NewContainer(),
		NonSwaggerRoutes: PathRecorderMux{
			mux: mux,
		},
		UnlistedRoutes: mux,
	}
	c.Container.ServeMux = mux
	c.Container.Router(restful.CurlyRouter{}) // e.g. for proxy/{kind}/{name}/{*}
	c.Container.RecoverHandler(func(panicReason interface{}, httpWriter http.ResponseWriter) {
		logStackOnRecover(s, panicReason, httpWriter)
	})
	c.Container.ServiceErrorHandler(func(serviceErr restful.ServiceError, request *restful.Request, response *restful.Response) {
		serviceErrorHandler(s, serviceErr, request, response)
	})

	return &c
}

//TODO: Unify with RecoverPanics?
func logStackOnRecover(s runtime.NegotiatedSerializer, panicReason interface{}, w http.ResponseWriter) {
	var buffer bytes.Buffer
	buffer.WriteString(fmt.Sprintf("recover from panic situation: - %v\r\n", panicReason))
	for i := 2; ; i++ {
		_, file, line, ok := rt.Caller(i)
		if !ok {
			break
		}
		buffer.WriteString(fmt.Sprintf("    %s:%d\r\n", file, line))
	}
	glog.Errorln(buffer.String())

	headers := http.Header{}
	if ct := w.Header().Get("Content-Type"); len(ct) > 0 {
		headers.Set("Accept", ct)
	}
	responsewriters.ErrorNegotiated(apierrors.NewGenericServerResponse(http.StatusInternalServerError, "", schema.GroupResource{}, "", "", 0, false), s, schema.GroupVersion{}, w, &http.Request{Header: headers})
}

func serviceErrorHandler(s runtime.NegotiatedSerializer, serviceErr restful.ServiceError, request *restful.Request, resp *restful.Response) {
	responsewriters.ErrorNegotiated(
		apierrors.NewGenericServerResponse(serviceErr.Code, "", schema.GroupResource{}, "", serviceErr.Message, 0, false),
		s,
		schema.GroupVersion{},
		resp,
		request.Request,
	)
}
