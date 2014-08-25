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

	"github.com/GoogleCloudPlatform/kubernetes/pkg/httplog"
)

type RedirectHandler struct {
	storage map[string]RESTStorage
	codec   Codec
}

func (r *RedirectHandler) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	parts := splitPath(req.URL.Path)
	if len(parts) != 2 || req.Method != "GET" {
		notFound(w, req)
		return
	}
	resourceName := parts[0]
	id := parts[1]
	storage, ok := r.storage[resourceName]
	if !ok {
		httplog.LogOf(w).Addf("'%v' has no storage object", resourceName)
		notFound(w, req)
		return
	}

	redirector, ok := storage.(Redirector)
	if !ok {
		httplog.LogOf(w).Addf("'%v' is not a redirector", resourceName)
		notFound(w, req)
		return
	}

	location, err := redirector.ResourceLocation(id)
	if err != nil {
		status := errToAPIStatus(err)
		writeJSON(status.Code, r.codec, status, w)
		return
	}

	w.Header().Set("Location", location)
	w.WriteHeader(http.StatusTemporaryRedirect)
}
