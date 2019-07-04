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
var _ = rest.GetterWithOptions(&LogREST{})

// New creates a new Pod log options object
func (r *LogREST) New() runtime.Object {
	// TODO - return a resource that represents a log
	return &api.Pod{}
}

// LogREST implements StorageMetadata
func (r *LogREST) ProducesMIMETypes(verb string) []string {
	// Since the default list does not include "plain/text", we need to
	// explicitly override ProducesMIMETypes, so that it gets added to
	// the "produces" section for pods/{name}/log
	return []string{
		"text/plain",
	}
}

// LogREST implements StorageMetadata, return string as the generating object
func (r *LogREST) ProducesObject(verb string) interface{} {
	return ""
}

// Get retrieves a runtime.Object that will stream the contents of the pod log
func (r *LogREST) Get(ctx context.Context, name string, opts runtime.Object) (runtime.Object, error) {
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
	return &genericrest.LocationStreamer{
		Location:        location,
		Transport:       transport,
		ContentType:     "text/plain",
		Flush:           logOpts.Follow,
		ResponseChecker: genericrest.NewGenericHttpResponseChecker(api.Resource("pods/log"), name),
		RedirectChecker: genericrest.PreventRedirects,
	}, nil
}

// NewGetOptions creates a new options object
func (r *LogREST) NewGetOptions() (runtime.Object, bool, string) {
	return &api.PodLogOptions{}, false, ""
}

// OverrideMetricsVerb override the GET verb to CONNECT for pod log resource
func (r *LogREST) OverrideMetricsVerb(oldVerb string) (newVerb string) {
	newVerb = oldVerb

	if oldVerb == "GET" {
		newVerb = "CONNECT"
	}

	return
}
