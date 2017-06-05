/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package rest

import (
	"net/http"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/rest"
	"k8s.io/kubernetes/pkg/runtime"
)

// HookHandler is a Kubernetes API compatible webhook that is able to get access to the raw request
// and response. Used when adapting existing webhook code to the Kubernetes patterns.
type HookHandler interface {
	ServeHTTP(w http.ResponseWriter, r *http.Request, ctx api.Context, name, subpath string) error
}

type httpHookHandler struct {
	http.Handler
}

func (h httpHookHandler) ServeHTTP(w http.ResponseWriter, r *http.Request, ctx api.Context, name, subpath string) error {
	h.Handler.ServeHTTP(w, r)
	return nil
}

// WebHook provides a reusable rest.Storage implementation for linking a generic WebHook handler
// into the Kube API pattern. It is intended to be used with GET or POST against a resource's
// named path, possibly as a subresource. The handler has access to the extracted information
// from the Kube apiserver including the context, the name, and the subpath.
type WebHook struct {
	h        HookHandler
	allowGet bool
}

var _ rest.Connecter = &WebHook{}

// NewWebHook creates an adapter that implements rest.Connector for the given HookHandler.
func NewWebHook(handler HookHandler, allowGet bool) *WebHook {
	return &WebHook{
		h:        handler,
		allowGet: allowGet,
	}
}

// NewHTTPWebHook creates an adapter that implements rest.Connector for the given http.Handler.
func NewHTTPWebHook(handler http.Handler, allowGet bool) *WebHook {
	return &WebHook{
		h:        httpHookHandler{handler},
		allowGet: allowGet,
	}
}

// New() responds with the status object.
func (h *WebHook) New() runtime.Object {
	return &api.Status{}
}

// Connect responds to connections with a ConnectHandler
func (h *WebHook) Connect(ctx api.Context, name string, options runtime.Object) (rest.ConnectHandler, error) {
	return &WebHookHandler{
		handler: h.h,
		ctx:     ctx,
		name:    name,
		options: options.(*api.PodProxyOptions),
	}, nil
}

// NewConnectionOptions identifies the options that should be passed to this hook
func (h *WebHook) NewConnectOptions() (runtime.Object, bool, string) {
	return &api.PodProxyOptions{}, true, "path"
}

// ConnectMethods returns the supported web hook types.
func (h *WebHook) ConnectMethods() []string {
	if h.allowGet {
		return []string{"GET", "POST"}
	}
	return []string{"POST"}
}

// WebHookHandler responds to web hook requests from the master.
type WebHookHandler struct {
	handler HookHandler
	ctx     api.Context
	name    string
	options *api.PodProxyOptions
	err     error
}

var _ rest.ConnectHandler = &WebHookHandler{}

func (h *WebHookHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	h.err = h.handler.ServeHTTP(w, r, h.ctx, h.name, h.options.Path)
	if h.err == nil {
		w.WriteHeader(http.StatusOK)
	}
}

func (h *WebHookHandler) RequestError() error {
	return h.err
}
