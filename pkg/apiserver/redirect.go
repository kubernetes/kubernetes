/*
Copyright 2014 Google Inc. All rights reserved.

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
	"strconv"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/httplog"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"

	"github.com/prometheus/client_golang/prometheus"
)

var (
	redirectCounter = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "apiserver_redirect_count",
			Help: "Counter of redirect requests broken out by apiserver resource and HTTP response code.",
		},
		[]string{"resource", "code"},
	)
)

func init() {
	prometheus.MustRegister(redirectCounter)
}

type RedirectHandler struct {
	storage                map[string]RESTStorage
	codec                  runtime.Codec
	apiRequestInfoResolver *APIRequestInfoResolver
}

func (r *RedirectHandler) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	apiResource := ""
	var httpCode int
	reqStart := time.Now()
	defer func() {
		redirectCounter.WithLabelValues(apiResource, strconv.Itoa(httpCode)).Inc()
		apiserverLatencies.WithLabelValues("redirect", "get", strconv.Itoa(httpCode)).Observe(float64((time.Since(reqStart)) / time.Microsecond))
	}()

	requestInfo, err := r.apiRequestInfoResolver.GetAPIRequestInfo(req)
	if err != nil {
		notFound(w, req)
		httpCode = http.StatusNotFound
		return
	}
	resource, parts := requestInfo.Resource, requestInfo.Parts
	ctx := api.WithNamespace(api.NewContext(), requestInfo.Namespace)

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
		apiResource = "invalidResource"
		httpCode = http.StatusNotFound
		return
	}
	apiResource = resource

	redirector, ok := storage.(Redirector)
	if !ok {
		httplog.LogOf(req, w).Addf("'%v' is not a redirector", resource)
		httpCode = errorJSON(errors.NewMethodNotSupported(resource, "redirect"), r.codec, w)
		return
	}

	location, err := redirector.ResourceLocation(ctx, id)
	if err != nil {
		status := errToAPIStatus(err)
		writeJSON(status.Code, r.codec, status, w)
		httpCode = status.Code
		return
	}

	w.Header().Set("Location", location)
	w.WriteHeader(http.StatusTemporaryRedirect)
	httpCode = http.StatusTemporaryRedirect
}
