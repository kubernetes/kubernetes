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

package rest

import (
	"fmt"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/rest"
	"k8s.io/kubernetes/pkg/api/validation"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	etcdgeneric "k8s.io/kubernetes/pkg/registry/generic/etcd"
	genericrest "k8s.io/kubernetes/pkg/registry/generic/rest"
	"k8s.io/kubernetes/pkg/registry/pod"
	"k8s.io/kubernetes/pkg/runtime"
)

// LogREST implements the log endpoint for a Pod
type LogREST struct {
	Store       *etcdgeneric.Etcd
	KubeletConn client.ConnectionInfoGetter
}

// LogREST implements GetterWithOptions
var _ = rest.GetterWithOptions(&LogREST{})

// New creates a new Pod log options object
func (r *LogREST) New() runtime.Object {
	// TODO - return a resource that represents a log
	return &api.Pod{}
}

// Get retrieves a runtime.Object that will stream the contents of the pod log
func (r *LogREST) Get(ctx api.Context, name string, opts runtime.Object) (runtime.Object, error) {
	logOpts, ok := opts.(*api.PodLogOptions)
	if !ok {
		return nil, fmt.Errorf("Invalid options object: %#v", opts)
	}
	if errs := validation.ValidatePodLogOptions(logOpts); len(errs) > 0 {
		return nil, errors.NewInvalid("podlogs", name, errs)
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
		ResponseChecker: genericrest.NewGenericHttpResponseChecker("Pod", name),
	}, nil
}

// NewGetOptions creates a new options object
func (r *LogREST) NewGetOptions() (runtime.Object, bool, string) {
	return &api.PodLogOptions{}, false, ""
}
