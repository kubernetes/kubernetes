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

package rest

import (
	"context"
	"fmt"
	"net/http"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	genericregistry "k8s.io/apiserver/pkg/registry/generic/registry"
	genericrest "k8s.io/apiserver/pkg/registry/generic/rest"
	"k8s.io/apiserver/pkg/registry/rest"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/core/validation"
	"k8s.io/kubernetes/pkg/kubelet/client"
	"k8s.io/kubernetes/pkg/registry/core/pod"
)

// LogREST implements the log endpoint for a Pod
type LogREST struct {
	KubeletConn client.ConnectionInfoGetter
	Store       *genericregistry.Store
}

// LogREST implements GetterWithOptions
var _ = rest.Connecter(&LogREST{})

// New returns an empty podLogOptions object.
func (r *LogREST) New() runtime.Object {
	return &api.PodLogOptions{}
}

// ConnectMethods returns the list of HTTP methods that supported
func (r *LogREST) ConnectMethods() []string {
	return []string{"GET"}
}

// NewConnectOptions returns versioned resource that represents logs parameters
func (r *LogREST) NewConnectOptions() (runtime.Object, bool, string) {
	return &api.PodLogOptions{}, false, ""
}

// Connect returns a handler for the pod logs
func (r *LogREST) Connect(ctx context.Context, name string, opts runtime.Object, responder rest.Responder) (http.Handler, error) {
	logOpts, ok := opts.(*api.PodLogOptions)
	if !ok {
		return nil, fmt.Errorf("invalid options object: %#v", opts)
	}
	if errs := validation.ValidatePodLogOptions(logOpts); len(errs) > 0 {
		return nil, errors.NewInvalid(api.Kind("PodLogOptions"), name, errs)
	}
	location, transport, err := pod.LogLocation(r.Store, r.KubeletConn, ctx, name, logOpts)
	if err != nil {
		return nil, err
	}
	streamer := &genericrest.LocationStreamer{
		Location:        location,
		Transport:       transport,
		ContentType:     "text/plain",
		Flush:           logOpts.Follow,
		ResponseChecker: genericrest.NewGenericHttpResponseChecker(api.Resource("pods/log"), name),
		RedirectChecker: genericrest.PreventRedirects,
	}
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		responder.Object(http.StatusOK, streamer)
	}), nil
}
