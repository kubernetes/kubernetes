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

package etcd

import (
	"fmt"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	etcderr "k8s.io/kubernetes/pkg/api/errors/etcd"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/registry/endpoint"
	"k8s.io/kubernetes/pkg/registry/pod"
	"k8s.io/kubernetes/pkg/storage"
	"k8s.io/kubernetes/pkg/watch"
)

const (
	// ServicePath is the path to service resources in etcd
	ServicePath string = "/services/specs"
)

// TODO(wojtek-t): Change it to use rest.StandardStorage (as everything else)
// and move it to service/ directory.

// TODO: Need to add a reconciler loop that makes sure that things in pods are reflected into
//       kubelet (and vice versa)

// Registry implements BindingRegistry, ControllerRegistry, EndpointRegistry,
// MinionRegistry, PodRegistry and ServiceRegistry, backed by etcd.
type Registry struct {
	storage.Interface
	pods      pod.Registry
	endpoints endpoint.Registry
}

// NewRegistry creates an etcd registry.
func NewRegistry(storage storage.Interface, pods pod.Registry, endpoints endpoint.Registry) *Registry {
	registry := &Registry{
		Interface: storage,
		pods:      pods,
		endpoints: endpoints,
	}
	return registry
}

// MakeEtcdListKey constructs etcd paths to resource directories enforcing namespace rules
func MakeEtcdListKey(ctx api.Context, prefix string) string {
	key := prefix
	ns, ok := api.NamespaceFrom(ctx)
	if ok && len(ns) > 0 {
		key = key + "/" + ns
	}
	return key
}

// MakeEtcdItemKey constructs etcd paths to a resource relative to prefix enforcing namespace rules.  If no namespace is on context, it errors.
func MakeEtcdItemKey(ctx api.Context, prefix string, id string) (string, error) {
	key := MakeEtcdListKey(ctx, prefix)
	ns, ok := api.NamespaceFrom(ctx)
	if !ok || len(ns) == 0 {
		return "", fmt.Errorf("invalid request.  Namespace parameter required.")
	}
	if len(id) == 0 {
		return "", fmt.Errorf("invalid request.  Id parameter required.")
	}
	key = key + "/" + id
	return key, nil
}

// makePodListKey constructs etcd paths to service directories enforcing namespace rules.
func makeServiceListKey(ctx api.Context) string {
	return MakeEtcdListKey(ctx, ServicePath)
}

// makeServiceKey constructs etcd paths to service items enforcing namespace rules.
func makeServiceKey(ctx api.Context, name string) (string, error) {
	return MakeEtcdItemKey(ctx, ServicePath, name)
}

// ListServices obtains a list of Services.
func (r *Registry) ListServices(ctx api.Context) (*api.ServiceList, error) {
	list := &api.ServiceList{}
	err := r.List(makeServiceListKey(ctx), list)
	return list, err
}

// CreateService creates a new Service.
func (r *Registry) CreateService(ctx api.Context, svc *api.Service) (*api.Service, error) {
	key, err := makeServiceKey(ctx, svc.Name)
	if err != nil {
		return nil, err
	}
	out := &api.Service{}
	err = r.Create(key, svc, out, 0)
	return out, etcderr.InterpretCreateError(err, "service", svc.Name)
}

// GetService obtains a Service specified by its name.
func (r *Registry) GetService(ctx api.Context, name string) (*api.Service, error) {
	key, err := makeServiceKey(ctx, name)
	if err != nil {
		return nil, err
	}
	var svc api.Service
	err = r.Get(key, &svc, false)
	if err != nil {
		return nil, etcderr.InterpretGetError(err, "service", name)
	}
	return &svc, nil
}

// DeleteService deletes a Service specified by its name.
func (r *Registry) DeleteService(ctx api.Context, name string) error {
	key, err := makeServiceKey(ctx, name)
	if err != nil {
		return err
	}
	err = r.RecursiveDelete(key, true)
	if err != nil {
		return etcderr.InterpretDeleteError(err, "service", name)
	}

	// TODO: can leave dangling endpoints, and potentially return incorrect
	// endpoints if a new service is created with the same name
	err = r.endpoints.DeleteEndpoints(ctx, name)
	if err != nil && !errors.IsNotFound(err) {
		return err
	}
	return nil
}

// UpdateService replaces an existing Service.
func (r *Registry) UpdateService(ctx api.Context, svc *api.Service) (*api.Service, error) {
	key, err := makeServiceKey(ctx, svc.Name)
	if err != nil {
		return nil, err
	}
	out := &api.Service{}
	err = r.Set(key, svc, out, 0)
	return out, etcderr.InterpretUpdateError(err, "service", svc.Name)
}

// WatchServices begins watching for new, changed, or deleted service configurations.
func (r *Registry) WatchServices(ctx api.Context, label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error) {
	version, err := storage.ParseWatchResourceVersion(resourceVersion, "service")
	if err != nil {
		return nil, err
	}
	if !label.Empty() {
		return nil, fmt.Errorf("label selectors are not supported on services")
	}
	if value, found := field.RequiresExactMatch("name"); found {
		key, err := makeServiceKey(ctx, value)
		if err != nil {
			return nil, err
		}
		// TODO: use generic.SelectionPredicate
		return r.Watch(key, version, storage.Everything)
	}
	if field.Empty() {
		return r.WatchList(makeServiceListKey(ctx), version, storage.Everything)
	}
	return nil, fmt.Errorf("only the 'name' and default (everything) field selectors are supported")
}
