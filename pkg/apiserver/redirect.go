/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"net/http"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/rest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/httplog"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

type RedirectHandler struct {
	storage                map[string]rest.Storage
	codec                  runtime.Codec
	context                api.RequestContextMapper
	apiRequestInfoResolver *APIRequestInfoResolver
}

func (r *RedirectHandler) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	var verb string
	var apiResource string
	var httpCode int
	reqStart := time.Now()
	defer monitor(&verb, &apiResource, util.GetClient(req), &httpCode, reqStart)

	requestInfo, err := r.apiRequestInfoResolver.GetAPIRequestInfo(req)
	if err != nil {
		notFound(w, req)
		httpCode = http.StatusNotFound
		return
	}
	verb = requestInfo.Verb
	resource, parts := requestInfo.Resource, requestInfo.Parts
	ctx, ok := r.context.Get(req)
	if !ok {
		ctx = api.NewContext()
	}
	ctx = api.WithNamespace(ctx, requestInfo.Namespace)

	// redirection requires /resource/resourceName path parts
	if len(parts) != 2 || req.Method != "GET" {
		notFound(w, req)
		httpCode = http.StatusNotFound
		return
	}
	id := parts[1]
	storage, ok := r.storage[resource]
	if !ok {
		httplog.LogOf(req, w).Addf("'%v' has no storage object", resource)
		notFound(w, req)
		httpCode = http.StatusNotFound
		return
	}
	apiResource = resource

	redirector, ok := storage.(rest.Redirector)
	if !ok {
		httplog.LogOf(req, w).Addf("'%v' is not a redirector", resource)
		httpCode = errorJSON(errors.NewMethodNotSupported(resource, "redirect"), r.codec, w)
		return
	}

	location, _, err := redirector.ResourceLocation(ctx, id)
	if err != nil {
		status := errToAPIStatus(err)
		writeJSON(status.Code, r.codec, status, w, true)
		httpCode = status.Code
		return
	}
	if location == nil {
		httplog.LogOf(req, w).Addf("ResourceLocation for %v returned nil", id)
		notFound(w, req)
		httpCode = http.StatusNotFound
		return
	}

	// Default to http
	if location.Scheme == "" {
		location.Scheme = "http"
	}

	w.Header().Set("Location", location.String())
	w.WriteHeader(http.StatusTemporaryRedirect)
	httpCode = http.StatusTemporaryRedirect
}
