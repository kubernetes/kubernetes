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

package apiserver

import (
	"github.com/emicklei/go-restful"

	"k8s.io/kubernetes/pkg/api"
	apierrors "k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/runtime"
)

func InstallServiceErrorHandler(s runtime.NegotiatedSerializer, container *restful.Container) {
	container.ServiceErrorHandler(func(serviceErr restful.ServiceError, request *restful.Request, response *restful.Response) {
		serviceErrorHandler(s, serviceErr, request, response)
	})
}

func serviceErrorHandler(s runtime.NegotiatedSerializer, serviceErr restful.ServiceError, request *restful.Request, response *restful.Response) {
	errorNegotiated(
		apierrors.NewGenericServerResponse(serviceErr.Code, "", api.Resource(""), "", serviceErr.Message, 0, false),
		s,
		unversioned.GroupVersion{},
		response.ResponseWriter,
		request.Request,
	)
}
