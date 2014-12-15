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
	path            string
}

// NewRegistry creates an etcd registry.
func NewRegistry(helper tools.EtcdHelper, boundPodFactory pod.BoundPodFactory) *Registry {
	registry := &Registry{
		EtcdHelper: helper,
	}
	registry.boundPodFactory = boundPodFactory
	return registry
}

// New way.
func NewRegistry2(helper tools.EtcdHelper, path string, boundPodFactory pod.BoundPodFactory) *Registry {
	reg := NewRegistry(helper, boundPodFactory)
	reg.path = path
	return reg
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

// makePodListKey constructs etcd paths to pod directories enforcing namespace rules.
func makePodListKey(ctx api.Context) string {
	return MakeEtcdListKey(ctx, PodPath)
}

// makePodKey constructs etcd paths to pod items enforcing namespace rules.
func makePodKey(ctx api.Context, id string) (string, error) {
	return MakeEtcdItemKey(ctx, PodPath, id)
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
			filtered = append(filtered, pod)
		}
	}
	allPods.Items = filtered
	return &allPods, nil
}

// WatchPods begins watching for new, changed, or deleted pods.
func (r *Registry) WatchPods(ctx api.Context, label, field labels.Selector, resourceVersion string) (watch.Interface, error) {
	version, err := tools.ParseWatchResourceVersion(resourceVersion, "pod")
	if err != nil {
		return nil, err
	}
	key := makePodListKey(ctx)
	return r.WatchList(key, version, func(obj runtime.Object) bool {
		podObj, ok := obj.(*api.Pod)
		if !ok {
			// Must be an error: return true to propagate to upper level.
			return true
		}
		fields := pod.PodToSelectableFields(podObj)
		return label.Matches(labels.Set(podObj.Labels)) && field.Matches(fields)
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
	return &pod, nil
}

func makeBoundPodsKey(machine string) string {
	return "/registry/nodes/" + machine + "/boundpods"
}

// CreatePod creates a pod based on a specification.
func (r *Registry) CreatePod(ctx api.Context, pod *api.Pod) error {
	// Set current status to "Waiting".
	pod.Status.Phase = api.PodPending
	pod.Status.Host = ""
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
		if pod.Status.Host != oldMachine {
			return nil, fmt.Errorf("pod %v is already assigned to host %v", pod.Name, pod.Status.Host)
		}
		pod.Status.Host = machine
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
	contKey := makeBoundPodsKey(machine)
	err = r.AtomicUpdate(contKey, &api.BoundPods{}, func(in runtime.Object) (runtime.Object, error) {
		boundPodList := in.(*api.BoundPods)
		boundPodList.Items = append(boundPodList.Items, *boundPod)
		if !constraint.Allowed(boundPodList.Items) {
			return nil, fmt.Errorf("the assignment would cause a constraint violation")
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
	scheduled := podOut.Status.Host != ""
	if scheduled {
		pod.Status.Host = podOut.Status.Host
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

	containerKey := makeBoundPodsKey(podOut.Status.Host)
	return r.AtomicUpdate(containerKey, &api.BoundPods{}, func(in runtime.Object) (runtime.Object, error) {
		boundPods := in.(*api.BoundPods)
		for ix := range boundPods.Items {
			if boundPods.Items[ix].Name == pod.Name {
				boundPods.Items[ix].Spec = pod.Spec
				return boundPods, nil
			}
		}
		// This really shouldn't happen
		glog.Warningf("Couldn't find: %s in %#v", pod.Name, boundPods)
		return boundPods, fmt.Errorf("failed to update pod, couldn't find %s in %#v", pod.Name, boundPods)
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
	machine := pod.Status.Host
	if machine == "" {
		// Pod was never scheduled anywhere, just return.
		return nil
	}
	// Next, remove the pod from the machine atomically.
	contKey := makeBoundPodsKey(machine)
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
func (r *Registry) WatchControllers(ctx api.Context, label, field labels.Selector, resourceVersion string) (watch.Interface, error) {
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
			pods, _ := r.ListPods(ctx, labels.Set(controller.Spec.Selector).AsSelector())
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
	version, err := tools.ParseWatchResourceVersion(resourceVersion, "endpoints")
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

func (r *Registry) ListMinions(ctx api.Context) (*api.NodeList, error) {
	minions := &api.NodeList{}
	err := r.ExtractToList("/registry/minions", minions)
	return minions, err
}

func (r *Registry) CreateMinion(ctx api.Context, minion *api.Node) error {
	// TODO: Add some validations.
	err := r.CreateObj(makeMinionKey(minion.Name), minion, 0)
	return etcderr.InterpretCreateError(err, "minion", minion.Name)
}

func (r *Registry) UpdateMinion(ctx api.Context, minion *api.Node) error {
	// TODO: Add some validations.
	err := r.SetObj(makeMinionKey(minion.Name), minion)
	return etcderr.InterpretUpdateError(err, "minion", minion.Name)
}

func (r *Registry) GetMinion(ctx api.Context, minionID string) (*api.Node, error) {
	var minion api.Node
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

// New way.
func (r *Registry) List(ctx api.Context, out runtime.Object) error {
	key := MakeEtcdListKey(ctx, r.path)
	return r.ExtractToList(key, out)
}
func (r *Registry) Get(ctx api.Context, name string, out runtime.Object) error {
	key, err := MakeEtcdItemKey(ctx, r.path, name)
	if err != nil {
		return err
	}
	err = r.ExtractObj(key, out, false)
	if err != nil {
		return etcderr.InterpretGetError(err, r.path, name)
	}
	return nil
}
func (r *Registry) Update(ctx api.Context, name string, obj runtime.Object) error {
	key, err := MakeEtcdItemKey(ctx, r.path, name)
	if err != nil {
		return err
	}
	// TODO: this is a really bad misuse of AtomicUpdate, need to compute a diff inside the loop.
	err = r.AtomicUpdate(key, obj,
		func(input runtime.Object) (runtime.Object, error) {
			// TODO: racy - label query is returning different results for two simultaneous updaters
			return obj, nil
		})
	return etcderr.InterpretUpdateError(err, r.path, name)
}

/* FIXME: not done yet
func (r *Registry) Watch(ctx api.Context, labels, fields labels.Selector, resourceVersion string) (watch.Interface, error) {
	//FIXME:
	panic("GenericWatch")
}
*/
