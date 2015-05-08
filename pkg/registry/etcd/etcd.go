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

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	etcderr "github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors/etcd"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/endpoint"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/pod"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/tools"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"

	"github.com/golang/glog"
)

const (
	// ControllerPath is the path to controller resources in etcd
	ControllerPath string = "/controllers"
	// ServicePath is the path to service resources in etcd
	ServicePath string = "/services/specs"
)

// TODO: Need to add a reconciler loop that makes sure that things in pods are reflected into
//       kubelet (and vice versa)

// Registry implements BindingRegistry, ControllerRegistry, EndpointRegistry,
// MinionRegistry, PodRegistry and ServiceRegistry, backed by etcd.
type Registry struct {
	tools.EtcdHelper
	pods      pod.Registry
	endpoints endpoint.Registry
}

// NewRegistry creates an etcd registry.
func NewRegistry(helper tools.EtcdHelper, pods pod.Registry, endpoints endpoint.Registry) *Registry {
	registry := &Registry{
		EtcdHelper: helper,
		pods:       pods,
		endpoints:  endpoints,
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

// ListControllers obtains a list of ReplicationControllers.
func (r *Registry) ListControllers(ctx api.Context) (*api.ReplicationControllerList, error) {
	controllers := &api.ReplicationControllerList{}
	key := makeControllerListKey(ctx)
	err := r.ExtractToList(key, controllers)
	return controllers, err
}

// WatchControllers begins watching for new, changed, or deleted controllers.
func (r *Registry) WatchControllers(ctx api.Context, label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error) {
	if !field.Empty() {
		return nil, fmt.Errorf("field selectors are not supported on replication controllers")
	}
	version, err := tools.ParseWatchResourceVersion(resourceVersion, "replicationControllers")
	if err != nil {
		return nil, err
	}
	key := makeControllerListKey(ctx)
	return r.WatchList(key, version, func(obj runtime.Object) bool {
		controller, ok := obj.(*api.ReplicationController)
		if !ok {
			// Must be an error: return true to propagate to upper level.
			return true
		}
		match := label.Matches(labels.Set(controller.Labels))
		if match {
			pods, err := r.pods.ListPods(ctx, labels.Set(controller.Spec.Selector).AsSelector())
			if err != nil {
				glog.Warningf("Error listing pods: %v", err)
				// No object that's useable so drop it on the floor
				return false
			}
			if pods == nil {
				glog.Warningf("Pods list is nil.  This should never happen...")
				// No object that's useable so drop it on the floor
				return false
			}
			controller.Status.Replicas = len(pods.Items)
		}
		return match
	})
}

// makeControllerListKey constructs etcd paths to controller directories enforcing namespace rules.
func makeControllerListKey(ctx api.Context) string {
	return MakeEtcdListKey(ctx, ControllerPath)
}

// makeControllerKey constructs etcd paths to controller items enforcing namespace rules.
func makeControllerKey(ctx api.Context, id string) (string, error) {
	return MakeEtcdItemKey(ctx, ControllerPath, id)
}

// GetController gets a specific ReplicationController specified by its ID.
func (r *Registry) GetController(ctx api.Context, controllerID string) (*api.ReplicationController, error) {
	var controller api.ReplicationController
	key, err := makeControllerKey(ctx, controllerID)
	if err != nil {
		return nil, err
	}
	err = r.ExtractObj(key, &controller, false)
	if err != nil {
		return nil, etcderr.InterpretGetError(err, "replicationController", controllerID)
	}
	return &controller, nil
}

// CreateController creates a new ReplicationController.
func (r *Registry) CreateController(ctx api.Context, controller *api.ReplicationController) (*api.ReplicationController, error) {
	key, err := makeControllerKey(ctx, controller.Name)
	if err != nil {
		return nil, err
	}
	out := &api.ReplicationController{}
	err = r.CreateObj(key, controller, out, 0)
	return out, etcderr.InterpretCreateError(err, "replicationController", controller.Name)
}

// UpdateController replaces an existing ReplicationController.
func (r *Registry) UpdateController(ctx api.Context, controller *api.ReplicationController) (*api.ReplicationController, error) {
	key, err := makeControllerKey(ctx, controller.Name)
	if err != nil {
		return nil, err
	}
	out := &api.ReplicationController{}
	err = r.SetObj(key, controller, out, 0)
	return out, etcderr.InterpretUpdateError(err, "replicationController", controller.Name)
}

// DeleteController deletes a ReplicationController specified by its ID.
func (r *Registry) DeleteController(ctx api.Context, controllerID string) error {
	key, err := makeControllerKey(ctx, controllerID)
	if err != nil {
		return err
	}
	err = r.Delete(key, false)
	return etcderr.InterpretDeleteError(err, "replicationController", controllerID)
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
	err := r.ExtractToList(makeServiceListKey(ctx), list)
	return list, err
}

// CreateService creates a new Service.
func (r *Registry) CreateService(ctx api.Context, svc *api.Service) (*api.Service, error) {
	key, err := makeServiceKey(ctx, svc.Name)
	if err != nil {
		return nil, err
	}
	out := &api.Service{}
	err = r.CreateObj(key, svc, out, 0)
	return out, etcderr.InterpretCreateError(err, "service", svc.Name)
}

// GetService obtains a Service specified by its name.
func (r *Registry) GetService(ctx api.Context, name string) (*api.Service, error) {
	key, err := makeServiceKey(ctx, name)
	if err != nil {
		return nil, err
	}
	var svc api.Service
	err = r.ExtractObj(key, &svc, false)
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
	err = r.Delete(key, true)
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
	err = r.SetObj(key, svc, out, 0)
	return out, etcderr.InterpretUpdateError(err, "service", svc.Name)
}

// WatchServices begins watching for new, changed, or deleted service configurations.
func (r *Registry) WatchServices(ctx api.Context, label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error) {
	version, err := tools.ParseWatchResourceVersion(resourceVersion, "service")
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
		return r.Watch(key, version, tools.Everything)
	}
	if field.Empty() {
		return r.WatchList(makeServiceListKey(ctx), version, tools.Everything)
	}
	return nil, fmt.Errorf("only the 'name' and default (everything) field selectors are supported")
}
