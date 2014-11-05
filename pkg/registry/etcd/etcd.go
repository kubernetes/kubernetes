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

package etcd

import (
	"fmt"
	"strconv"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	etcderr "github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors/etcd"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/validation"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/constraint"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/pod"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/tools"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"

	"github.com/golang/glog"
)

const (
	// PodPath is the path to pod resources in etcd
	PodPath string = "/registry/pods"
	// ControllerPath is the path to controller resources in etcd
	ControllerPath string = "/registry/controllers"
	// ServicePath is the path to service resources in etcd
	ServicePath string = "/registry/services/specs"
	// ServiceEndpointPath is the path to service endpoints resources in etcd
	ServiceEndpointPath string = "/registry/services/endpoints"
)

// TODO: Need to add a reconciler loop that makes sure that things in pods are reflected into
//       kubelet (and vice versa)

// Registry implements PodRegistry, ControllerRegistry, ServiceRegistry and MinionRegistry, backed by etcd.
type Registry struct {
	tools.EtcdHelper
	boundPodFactory pod.BoundPodFactory
}

// NewRegistry creates an etcd registry.
func NewRegistry(helper tools.EtcdHelper, boundPodFactory pod.BoundPodFactory) *Registry {
	registry := &Registry{
		EtcdHelper: helper,
	}
	registry.boundPodFactory = boundPodFactory
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
		return "", fmt.Errorf("Invalid request.  Namespace parameter required.")
	}
	if len(id) == 0 {
		return "", fmt.Errorf("Invalid request.  Id parameter required.")
	}
	key = key + "/" + id
	return key, nil
}

// makePodListKey constructs etcd paths to pod directories enforcing namespace rules.
func makePodListKey(ctx api.Context) string {
	return MakeEtcdListKey(ctx, PodPath)
}

// makePodKey constructs etcd paths to pod items enforcing namespace rules.
func makePodKey(ctx api.Context, id string) (string, error) {
	return MakeEtcdItemKey(ctx, PodPath, id)
}

// ParseWatchResourceVersion takes a resource version argument and converts it to
// the etcd version we should pass to helper.Watch(). Because resourceVersion is
// an opaque value, the default watch behavior for non-zero watch is to watch
// the next value (if you pass "1", you will see updates from "2" onwards).
func ParseWatchResourceVersion(resourceVersion, kind string) (uint64, error) {
	if resourceVersion == "" || resourceVersion == "0" {
		return 0, nil
	}
	version, err := strconv.ParseUint(resourceVersion, 10, 64)
	if err != nil {
		return 0, etcderr.InterpretResourceVersionError(err, kind, resourceVersion)
	}
	return version + 1, nil
}

// ListPods obtains a list of pods with labels that match selector.
func (r *Registry) ListPods(ctx api.Context, selector labels.Selector) (*api.PodList, error) {
	return r.ListPodsPredicate(ctx, func(pod *api.Pod) bool {
		return selector.Matches(labels.Set(pod.Labels))
	})
}

// ListPodsPredicate obtains a list of pods that match filter.
func (r *Registry) ListPodsPredicate(ctx api.Context, filter func(*api.Pod) bool) (*api.PodList, error) {
	allPods := api.PodList{}
	key := makePodListKey(ctx)
	err := r.ExtractToList(key, &allPods)
	if err != nil {
		return nil, err
	}
	filtered := []api.Pod{}
	for _, pod := range allPods.Items {
		if filter(&pod) {
			// TODO: Currently nothing sets CurrentState.Host. We need a feedback loop that sets
			// the CurrentState.Host and Status fields. Here we pretend that reality perfectly
			// matches our desires.
			pod.CurrentState.Host = pod.DesiredState.Host
			filtered = append(filtered, pod)
		}
	}
	allPods.Items = filtered
	return &allPods, nil
}

// WatchPods begins watching for new, changed, or deleted pods.
func (r *Registry) WatchPods(ctx api.Context, resourceVersion string, filter func(*api.Pod) bool) (watch.Interface, error) {
	version, err := ParseWatchResourceVersion(resourceVersion, "pod")
	if err != nil {
		return nil, err
	}
	key := makePodListKey(ctx)
	return r.WatchList(key, version, func(obj runtime.Object) bool {
		switch t := obj.(type) {
		case *api.Pod:
			return filter(t)
		default:
			// Must be an error
			return true
		}
	})
}

// GetPod gets a specific pod specified by its ID.
func (r *Registry) GetPod(ctx api.Context, id string) (*api.Pod, error) {
	var pod api.Pod
	key, err := makePodKey(ctx, id)
	if err != nil {
		return nil, err
	}
	if err = r.ExtractObj(key, &pod, false); err != nil {
		return nil, etcderr.InterpretGetError(err, "pod", id)
	}
	// TODO: Currently nothing sets CurrentState.Host. We need a feedback loop that sets
	// the CurrentState.Host and Status fields. Here we pretend that reality perfectly
	// matches our desires.
	pod.CurrentState.Host = pod.DesiredState.Host
	return &pod, nil
}

func makeContainerKey(machine string) string {
	return "/registry/nodes/" + machine + "/boundpods"
}

// CreatePod creates a pod based on a specification.
func (r *Registry) CreatePod(ctx api.Context, pod *api.Pod) error {
	// Set current status to "Waiting".
	pod.CurrentState.Status = api.PodPending
	pod.CurrentState.Host = ""
	// DesiredState.Host == "" is a signal to the scheduler that this pod needs scheduling.
	pod.DesiredState.Status = api.PodRunning
	pod.DesiredState.Host = ""
	key, err := makePodKey(ctx, pod.Name)
	if err != nil {
		return err
	}
	err = r.CreateObj(key, pod, 0)
	return etcderr.InterpretCreateError(err, "pod", pod.Name)
}

// ApplyBinding implements binding's registry
func (r *Registry) ApplyBinding(ctx api.Context, binding *api.Binding) error {
	return etcderr.InterpretCreateError(r.assignPod(ctx, binding.PodID, binding.Host), "binding", "")
}

// setPodHostTo sets the given pod's host to 'machine' iff it was previously 'oldMachine'.
// Returns the current state of the pod, or an error.
func (r *Registry) setPodHostTo(ctx api.Context, podID, oldMachine, machine string) (finalPod *api.Pod, err error) {
	podKey, err := makePodKey(ctx, podID)
	if err != nil {
		return nil, err
	}
	err = r.AtomicUpdate(podKey, &api.Pod{}, func(obj runtime.Object) (runtime.Object, error) {
		pod, ok := obj.(*api.Pod)
		if !ok {
			return nil, fmt.Errorf("unexpected object: %#v", obj)
		}
		if pod.DesiredState.Host != oldMachine {
			return nil, fmt.Errorf("pod %v is already assigned to host %v", pod.Name, pod.DesiredState.Host)
		}
		pod.DesiredState.Host = machine
		finalPod = pod
		return pod, nil
	})
	return finalPod, err
}

// assignPod assigns the given pod to the given machine.
func (r *Registry) assignPod(ctx api.Context, podID string, machine string) error {
	finalPod, err := r.setPodHostTo(ctx, podID, "", machine)
	if err != nil {
		return err
	}
	boundPod, err := r.boundPodFactory.MakeBoundPod(machine, finalPod)
	if err != nil {
		return err
	}
	// Doing the constraint check this way provides atomicity guarantees.
	contKey := makeContainerKey(machine)
	err = r.AtomicUpdate(contKey, &api.BoundPods{}, func(in runtime.Object) (runtime.Object, error) {
		boundPodList := in.(*api.BoundPods)
		boundPodList.Items = append(boundPodList.Items, *boundPod)
		if !constraint.Allowed(boundPodList.Items) {
			return nil, fmt.Errorf("The assignment would cause a constraint violation")
		}
		return boundPodList, nil
	})
	if err != nil {
		// Put the pod's host back the way it was. This is a terrible hack, but
		// can't really be helped, since there's not really a way to do atomic
		// multi-object changes in etcd.
		if _, err2 := r.setPodHostTo(ctx, podID, machine, ""); err2 != nil {
			glog.Errorf("Stranding pod %v; couldn't clear host after previous error: %v", podID, err2)
		}
	}
	return err
}

func (r *Registry) UpdatePod(ctx api.Context, pod *api.Pod) error {
	var podOut api.Pod
	podKey, err := makePodKey(ctx, pod.Name)
	if err != nil {
		return err
	}
	err = r.EtcdHelper.ExtractObj(podKey, &podOut, false)
	if err != nil {
		return err
	}
	scheduled := podOut.DesiredState.Host != ""
	if scheduled {
		pod.DesiredState.Host = podOut.DesiredState.Host
		// If it's already been scheduled, limit the types of updates we'll accept.
		errs := validation.ValidatePodUpdate(pod, &podOut)
		if len(errs) != 0 {
			return errors.NewInvalid("Pod", pod.Name, errs)
		}
	}
	// There's no race with the scheduler, because either this write will fail because the host
	// has been updated, or the host update will fail because this pod has been updated.
	err = r.EtcdHelper.SetObj(podKey, pod)
	if err != nil {
		return err
	}
	if !scheduled {
		// never scheduled, just update.
		return nil
	}
	containerKey := makeContainerKey(podOut.DesiredState.Host)
	return r.AtomicUpdate(containerKey, &api.ContainerManifestList{}, func(in runtime.Object) (runtime.Object, error) {
		manifests := in.(*api.ContainerManifestList)
		for ix := range manifests.Items {
			if manifests.Items[ix].ID == pod.Name {
				manifests.Items[ix] = pod.DesiredState.Manifest
				return manifests, nil
			}
		}
		// This really shouldn't happen
		glog.Warningf("Couldn't find: %s in %#v", pod.Name, manifests)
		return manifests, fmt.Errorf("Failed to update pod, couldn't find %s in %#v", pod.Name, manifests)
	})
}

// DeletePod deletes an existing pod specified by its ID.
func (r *Registry) DeletePod(ctx api.Context, podID string) error {
	var pod api.Pod
	podKey, err := makePodKey(ctx, podID)
	if err != nil {
		return err
	}
	err = r.ExtractObj(podKey, &pod, false)
	if err != nil {
		return etcderr.InterpretDeleteError(err, "pod", podID)
	}
	// First delete the pod, so a scheduler doesn't notice it getting removed from the
	// machine and attempt to put it somewhere.
	err = r.Delete(podKey, true)
	if err != nil {
		return etcderr.InterpretDeleteError(err, "pod", podID)
	}
	machine := pod.DesiredState.Host
	if machine == "" {
		// Pod was never scheduled anywhere, just return.
		return nil
	}
	// Next, remove the pod from the machine atomically.
	contKey := makeContainerKey(machine)
	return r.AtomicUpdate(contKey, &api.BoundPods{}, func(in runtime.Object) (runtime.Object, error) {
		pods := in.(*api.BoundPods)
		newPods := make([]api.BoundPod, 0, len(pods.Items))
		found := false
		for _, pod := range pods.Items {
			if pod.Name != podID {
				newPods = append(newPods, pod)
			} else {
				found = true
			}
		}
		if !found {
			// This really shouldn't happen, it indicates something is broken, and likely
			// there is a lost pod somewhere.
			// However it is "deleted" so log it and move on
			glog.Warningf("Couldn't find: %s in %#v", podID, pods)
		}
		pods.Items = newPods
		return pods, nil
	})
}

// ListControllers obtains a list of ReplicationControllers.
func (r *Registry) ListControllers(ctx api.Context) (*api.ReplicationControllerList, error) {
	controllers := &api.ReplicationControllerList{}
	key := makeControllerListKey(ctx)
	err := r.ExtractToList(key, controllers)
	return controllers, err
}

// WatchControllers begins watching for new, changed, or deleted controllers.
func (r *Registry) WatchControllers(ctx api.Context, resourceVersion string) (watch.Interface, error) {
	version, err := ParseWatchResourceVersion(resourceVersion, "replicationControllers")
	if err != nil {
		return nil, err
	}
	key := makeControllerListKey(ctx)
	return r.WatchList(key, version, tools.Everything)
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
func (r *Registry) CreateController(ctx api.Context, controller *api.ReplicationController) error {
	key, err := makeControllerKey(ctx, controller.Name)
	if err != nil {
		return err
	}
	err = r.CreateObj(key, controller, 0)
	return etcderr.InterpretCreateError(err, "replicationController", controller.Name)
}

// UpdateController replaces an existing ReplicationController.
func (r *Registry) UpdateController(ctx api.Context, controller *api.ReplicationController) error {
	key, err := makeControllerKey(ctx, controller.Name)
	if err != nil {
		return err
	}
	err = r.SetObj(key, controller)
	return etcderr.InterpretUpdateError(err, "replicationController", controller.Name)
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

// makePodKey constructs etcd paths to service items enforcing namespace rules.
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
func (r *Registry) CreateService(ctx api.Context, svc *api.Service) error {
	key, err := makeServiceKey(ctx, svc.Name)
	if err != nil {
		return err
	}
	err = r.CreateObj(key, svc, 0)
	return etcderr.InterpretCreateError(err, "service", svc.Name)
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

// GetEndpoints obtains the endpoints for the service identified by 'name'.
func (r *Registry) GetEndpoints(ctx api.Context, name string) (*api.Endpoints, error) {
	var endpoints api.Endpoints
	key, err := makeServiceEndpointsKey(ctx, name)
	if err != nil {
		return nil, err
	}
	err = r.ExtractObj(key, &endpoints, false)
	if err != nil {
		return nil, etcderr.InterpretGetError(err, "endpoints", name)
	}
	return &endpoints, nil
}

// makeServiceEndpointsListKey constructs etcd paths to service endpoint directories enforcing namespace rules.
func makeServiceEndpointsListKey(ctx api.Context) string {
	return MakeEtcdListKey(ctx, ServiceEndpointPath)
}

// makeServiceEndpointsListKey constructs etcd paths to service endpoint items enforcing namespace rules.
func makeServiceEndpointsKey(ctx api.Context, name string) (string, error) {
	return MakeEtcdItemKey(ctx, ServiceEndpointPath, name)
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
	key, err = makeServiceEndpointsKey(ctx, name)
	if err != nil {
		return err
	}
	if err := r.Delete(key, true); err != nil && !tools.IsEtcdNotFound(err) {
		return etcderr.InterpretDeleteError(err, "endpoints", name)
	}
	return nil
}

// UpdateService replaces an existing Service.
func (r *Registry) UpdateService(ctx api.Context, svc *api.Service) error {
	key, err := makeServiceKey(ctx, svc.Name)
	if err != nil {
		return err
	}
	err = r.SetObj(key, svc)
	return etcderr.InterpretUpdateError(err, "service", svc.Name)
}

// WatchServices begins watching for new, changed, or deleted service configurations.
func (r *Registry) WatchServices(ctx api.Context, label, field labels.Selector, resourceVersion string) (watch.Interface, error) {
	version, err := ParseWatchResourceVersion(resourceVersion, "service")
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
		return r.Watch(key, version), nil
	}
	if field.Empty() {
		return r.WatchList(makeServiceListKey(ctx), version, tools.Everything)
	}
	return nil, fmt.Errorf("only the 'name' and default (everything) field selectors are supported")
}

// ListEndpoints obtains a list of Services.
func (r *Registry) ListEndpoints(ctx api.Context) (*api.EndpointsList, error) {
	list := &api.EndpointsList{}
	key := makeServiceEndpointsListKey(ctx)
	err := r.ExtractToList(key, list)
	return list, err
}

// UpdateEndpoints update Endpoints of a Service.
func (r *Registry) UpdateEndpoints(ctx api.Context, endpoints *api.Endpoints) error {
	key, err := makeServiceEndpointsKey(ctx, endpoints.Name)
	if err != nil {
		return err
	}
	// TODO: this is a really bad misuse of AtomicUpdate, need to compute a diff inside the loop.
	err = r.AtomicUpdate(key, &api.Endpoints{},
		func(input runtime.Object) (runtime.Object, error) {
			// TODO: racy - label query is returning different results for two simultaneous updaters
			return endpoints, nil
		})
	return etcderr.InterpretUpdateError(err, "endpoints", endpoints.Name)
}

// WatchEndpoints begins watching for new, changed, or deleted endpoint configurations.
func (r *Registry) WatchEndpoints(ctx api.Context, label, field labels.Selector, resourceVersion string) (watch.Interface, error) {
	version, err := ParseWatchResourceVersion(resourceVersion, "endpoints")
	if err != nil {
		return nil, err
	}
	if !label.Empty() {
		return nil, fmt.Errorf("label selectors are not supported on endpoints")
	}
	if value, found := field.RequiresExactMatch("name"); found {
		key, err := makeServiceEndpointsKey(ctx, value)
		if err != nil {
			return nil, err
		}
		return r.Watch(key, version), nil
	}
	if field.Empty() {
		return r.WatchList(makeServiceEndpointsListKey(ctx), version, tools.Everything)
	}
	return nil, fmt.Errorf("only the 'ID' and default (everything) field selectors are supported")
}

func makeMinionKey(minionID string) string {
	return "/registry/minions/" + minionID
}

func (r *Registry) ListMinions(ctx api.Context) (*api.MinionList, error) {
	minions := &api.MinionList{}
	err := r.ExtractToList("/registry/minions", minions)
	return minions, err
}

func (r *Registry) CreateMinion(ctx api.Context, minion *api.Minion) error {
	// TODO: Add some validations.
	err := r.CreateObj(makeMinionKey(minion.Name), minion, 0)
	return etcderr.InterpretCreateError(err, "minion", minion.Name)
}

func (r *Registry) GetMinion(ctx api.Context, minionID string) (*api.Minion, error) {
	var minion api.Minion
	key := makeMinionKey(minionID)
	err := r.ExtractObj(key, &minion, false)
	if err != nil {
		return nil, etcderr.InterpretGetError(err, "minion", minion.Name)
	}
	return &minion, nil
}

func (r *Registry) DeleteMinion(ctx api.Context, minionID string) error {
	key := makeMinionKey(minionID)
	err := r.Delete(key, true)
	if err != nil {
		return etcderr.InterpretDeleteError(err, "minion", minionID)
	}
	return nil
}
