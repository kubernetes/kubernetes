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

package genericapiserver

import (
	"fmt"
	"net/http"

	"k8s.io/kubernetes/pkg/api"
	utilruntime "k8s.io/kubernetes/pkg/util/runtime"
)

type requestContextType int

const skipAuthContextKey requestContextType = iota

func withSkipAuth(parent api.Context) api.Context {
	return api.WithValue(parent, skipAuthContextKey, true)
}

func skipAuthFrom(ctx api.Context) bool {
	v := ctx.Value(skipAuthContextKey)
	return v != nil
}

func internalError(w http.ResponseWriter, req *http.Request, err error) {
	w.WriteHeader(http.StatusInternalServerError)
	fmt.Fprintf(w, "Internal Server Error: %#v", req.RequestURI)
	utilruntime.HandleError(err)
}
