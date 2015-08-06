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

package status

import (
	"fmt"
	"net/url"
	"strings"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/rest"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/master/ports"
	"k8s.io/kubernetes/pkg/probe"
	probehttp "k8s.io/kubernetes/pkg/probe/http"
	"k8s.io/kubernetes/pkg/registry/component"
	"k8s.io/kubernetes/pkg/runtime"
	db_storage "k8s.io/kubernetes/pkg/storage"
	etcd_storage "k8s.io/kubernetes/pkg/storage/etcd"
)

var (
	defaultControllerManagerEndpoint = url.URL{
		Scheme: "http",
		Host:   fmt.Sprintf("127.0.0.1:%d", ports.ControllerManagerPort),
		Path:   "/healthz",
	}

	defaultSchedulerEndpoint = url.URL{
		Scheme: "http",
		Host:   fmt.Sprintf("127.0.0.1:%d", ports.SchedulerPort),
		Path:   "/healthz",
	}
)

type storage struct {
	componentRegistry component.Registry
	databaseStorage   db_storage.Interface
	httpClient        probehttp.HTTPGetter
}

// NewStorage returns a new ReadableStorage that performs component health checks.
func NewStorage(
	componentRegistry component.Registry,
	databaseStorage db_storage.Interface,
	httpClient probehttp.HTTPGetter,
) rest.ReadableStorage {
	return &storage{
		componentRegistry: componentRegistry,
		databaseStorage:   databaseStorage,
		httpClient:        httpClient,
	}
}

// New returns an empty ComponentStatus.
// Satisfies the rest.Storage interface.
func (rs *storage) New() runtime.Object {
	return &api.ComponentStatus{}
}

// NewList returns an empty ComponentStatusList.
// Satisfies the rest.Lister interface.
func (rs *storage) NewList() runtime.Object {
	return &api.ComponentStatusList{}
}

func healthzURL(input *url.URL) *url.URL {
	path := input.Path
	if strings.HasPrefix(path, "/") {
		path += "/"
	}
	path += "healthz"
	return &url.URL{
		Scheme: input.Scheme,
		Host:   input.Host,
		Path:   path,
	}
}

func etcdHealthURL(input *url.URL) *url.URL {
	path := input.Path
	if strings.HasPrefix(path, "/") {
		path += "/"
	}
	path += "health"
	return &url.URL{
		Scheme: input.Scheme,
		Host:   input.Host,
		Path:   path,
	}
}

// List returns a list of statuses of the components that match the label and field selectors.
// Satisfies the rest.Lister interface.
func (rs *storage) List(ctx api.Context, labels labels.Selector, fields fields.Selector) (runtime.Object, error) {
	components, err := rs.componentRegistry.ListComponents(ctx, labels, fields)
	if err != nil {
		return nil, fmt.Errorf("failed to retrieve component status list: %s: ", err)
	}

	// TODO: This should be parallelized.
	results := []api.ComponentStatus{}
	for _, component := range components.Items {
		rootURL, err := url.Parse(component.URL)
		if err != nil {
			return nil, fmt.Errorf("failed to parse component url: %v", err)
		}
		status := rs.getComponentStatus(component.ObjectMeta.Name, probehttp.URLProber{
			URL:    healthzURL(rootURL),
			Client: rs.httpClient,
		})
		results = append(results, *status)
	}

	//TODO: remove defaults after component registration is implemented
	// use defaults if no components have registered
	if len(results) == 0 {
		status := rs.getComponentStatus("controller-manager", probehttp.URLProber{
			URL:    &defaultControllerManagerEndpoint,
			Client: rs.httpClient,
		})
		results = append(results, *status)

		status = rs.getComponentStatus("scheduler", probehttp.URLProber{
			URL:    &defaultSchedulerEndpoint,
			Client: rs.httpClient,
		})
		results = append(results, *status)
	}

	//TODO: move etcd health to a separate registry and /storage/status endpoint
	for ix, machine := range rs.databaseStorage.Backends() {
		name := fmt.Sprintf("etcd-%d", ix)
		rootURL, err := url.Parse(machine)
		if err != nil {
			return nil, fmt.Errorf("failed to parse etcd url: %v", err)
		}
		//TODO: set default port? host:4001 (undesirable if host is a proxy)
		status := rs.getComponentStatus(name, probehttp.URLProber{
			URL:        etcdHealthURL(rootURL),
			Client:     rs.httpClient,
			Validators: []probehttp.BodyValidator{etcd_storage.EtcdHealthValidator},
		})
		results = append(results, *status)
	}

	return &api.ComponentStatusList{Items: results}, nil
}

// Get finds the status of a component by name and returns it.
// Satisfies the rest.Getter interface.
func (rs *storage) Get(ctx api.Context, name string) (runtime.Object, error) {
	component, err := rs.componentRegistry.GetComponent(ctx, name)
	if err != nil {
		return nil, fmt.Errorf("failed to retrieve component status: %v", err)
	}

	rootURL, err := url.Parse(component.URL)
	if err != nil {
		return nil, fmt.Errorf("failed to parse component url: %v", err)
	}

	return rs.getComponentStatus(name, probehttp.URLProber{
		URL:    healthzURL(rootURL),
		Client: rs.httpClient,
	}), nil
}

func toConditionStatus(s probe.Result) api.ConditionStatus {
	switch s {
	case probe.Success:
		return api.ConditionTrue
	case probe.Failure:
		return api.ConditionFalse
	default:
		return api.ConditionUnknown
	}
}

func (rs *storage) getComponentStatus(name string, component probehttp.URLProber) *api.ComponentStatus {
	status, msg, err := component.Probe()
	var errorMsg string
	if err != nil {
		errorMsg = err.Error()
	} else {
		errorMsg = "nil"
	}

	c := &api.ComponentCondition{
		Type:    api.ComponentHealthy,
		Status:  toConditionStatus(status),
		Message: msg,
		Error:   errorMsg,
	}

	retVal := &api.ComponentStatus{
		Conditions: []api.ComponentCondition{*c},
	}
	retVal.Name = name

	return retVal
}
