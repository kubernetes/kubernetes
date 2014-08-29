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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/apiserver"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/constraint"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/tools"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"

	"github.com/golang/glog"
)

// TODO: Need to add a reconciler loop that makes sure that things in pods are reflected into
//       kubelet (and vice versa)

// Registry implements PodRegistry, ControllerRegistry and ServiceRegistry
// with backed by etcd.
type Registry struct {
	tools.EtcdHelper
	manifestFactory ManifestFactory
}

// NewRegistry creates an etcd registry.
func NewRegistry(client tools.EtcdClient) *Registry {
	registry := &Registry{
		EtcdHelper: tools.EtcdHelper{
			client,
			api.Codec,
			api.ResourceVersioner,
		},
	}
	registry.manifestFactory = &BasicManifestFactory{
		serviceRegistry: registry,
	}
	return registry
}

func makePodKey(podID string) string {
	return "/registry/pods/" + podID
}

// ListPods obtains a list of pods that match selector.
func (r *Registry) ListPods(selector labels.Selector) (*api.PodList, error) {
	allPods := api.PodList{}
	err := r.ExtractList("/registry/pods", &allPods.Items, &allPods.ResourceVersion)
	if err != nil {
		return nil, err
	}
	filtered := []api.Pod{}
	for _, pod := range allPods.Items {
		if selector.Matches(labels.Set(pod.Labels)) {
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
func (r *Registry) WatchPods(resourceVersion uint64, filter func(*api.Pod) bool) (watch.Interface, error) {
	return r.WatchList("/registry/pods", resourceVersion, func(obj interface{}) bool {
		pod, ok := obj.(*api.Pod)
		if !ok {
			glog.Errorf("Unexpected object during pod watch: %#v", obj)
			return false
		}
		return filter(pod)
	})
}

// GetPod gets a specific pod specified by its ID.
func (r *Registry) GetPod(podID string) (*api.Pod, error) {
	var pod api.Pod
	if err := r.ExtractObj(makePodKey(podID), &pod, false); err != nil {
		return nil, err
	}
	// TODO: Currently nothing sets CurrentState.Host. We need a feedback loop that sets
	// the CurrentState.Host and Status fields. Here we pretend that reality perfectly
	// matches our desires.
	pod.CurrentState.Host = pod.DesiredState.Host
	return &pod, nil
}

func makeContainerKey(machine string) string {
	return "/registry/hosts/" + machine + "/kubelet"
}

// CreatePod creates a pod based on a specification.
func (r *Registry) CreatePod(pod api.Pod) error {
	// Set current status to "Waiting".
	pod.CurrentState.Status = api.PodWaiting
	pod.CurrentState.Host = ""
	// DesiredState.Host == "" is a signal to the scheduler that this pod needs scheduling.
	pod.DesiredState.Status = api.PodRunning
	pod.DesiredState.Host = ""
	return r.CreateObj(makePodKey(pod.ID), &pod)
}

// ApplyBinding implements binding's registry
func (r *Registry) ApplyBinding(binding *api.Binding) error {
	return r.assignPod(binding.PodID, binding.Host)
}

// setPodHostTo sets the given pod's host to 'machine' iff it was previously 'oldMachine'.
// Returns the current state of the pod, or an error.
func (r *Registry) setPodHostTo(podID, oldMachine, machine string) (finalPod *api.Pod, err error) {
	podKey := makePodKey(podID)
	err = r.AtomicUpdate(podKey, &api.Pod{}, func(obj interface{}) (interface{}, error) {
		pod, ok := obj.(*api.Pod)
		if !ok {
			return nil, fmt.Errorf("unexpected object: %#v", obj)
		}
		if pod.DesiredState.Host != oldMachine {
			return nil, fmt.Errorf("pod %v is already assigned to host %v", pod.ID, pod.DesiredState.Host)
		}
		pod.DesiredState.Host = machine
		finalPod = pod
		return pod, nil
	})
	return finalPod, err
}

// assignPod assigns the given pod to the given machine.
func (r *Registry) assignPod(podID string, machine string) error {
	finalPod, err := r.setPodHostTo(podID, "", machine)
	if err != nil {
		return err
	}
	// TODO: move this to a watch/rectification loop.
	manifest, err := r.manifestFactory.MakeManifest(machine, *finalPod)
	if err != nil {
		return err
	}
	contKey := makeContainerKey(machine)
	err = r.AtomicUpdate(contKey, &api.ContainerManifestList{}, func(in interface{}) (interface{}, error) {
		manifests := *in.(*api.ContainerManifestList)
		manifests.Items = append(manifests.Items, manifest)
		if !constraint.Allowed(manifests.Items) {
			return nil, fmt.Errorf("The assignment would cause a constraint violation")
		}
		return manifests, nil
	})
	if err != nil {
		// Put the pod's host back the way it was. This is a terrible hack that
		// won't be needed if we convert this to a rectification loop.
		if _, err2 := r.setPodHostTo(podID, machine, ""); err2 != nil {
			glog.Errorf("Stranding pod %v; couldn't clear host after previous error: %v", podID, err2)
		}
	}
	return err
}

func (r *Registry) UpdatePod(pod api.Pod) error {
	return fmt.Errorf("unimplemented!")
}

// DeletePod deletes an existing pod specified by its ID.
func (r *Registry) DeletePod(podID string) error {
	var pod api.Pod
	podKey := makePodKey(podID)
	err := r.ExtractObj(podKey, &pod, false)
	if tools.IsEtcdNotFound(err) {
		return apiserver.NewNotFoundErr("pod", podID)
	}
	if err != nil {
		return err
	}
	// First delete the pod, so a scheduler doesn't notice it getting removed from the
	// machine and attempt to put it somewhere.
	err = r.Delete(podKey, true)
	if tools.IsEtcdNotFound(err) {
		return apiserver.NewNotFoundErr("pod", podID)
	}
	if err != nil {
		return err
	}
	machine := pod.DesiredState.Host
	if machine == "" {
		// Pod was never scheduled anywhere, just return.
		return nil
	}
	// Next, remove the pod from the machine atomically.
	contKey := makeContainerKey(machine)
	return r.AtomicUpdate(contKey, &api.ContainerManifestList{}, func(in interface{}) (interface{}, error) {
		manifests := in.(*api.ContainerManifestList)
		newManifests := make([]api.ContainerManifest, 0, len(manifests.Items))
		found := false
		for _, manifest := range manifests.Items {
			if manifest.ID != podID {
				newManifests = append(newManifests, manifest)
			} else {
				found = true
			}
		}
		if !found {
			// This really shouldn't happen, it indicates something is broken, and likely
			// there is a lost pod somewhere.
			// However it is "deleted" so log it and move on
			glog.Infof("Couldn't find: %s in %#v", podID, manifests)
		}
		manifests.Items = newManifests
		return manifests, nil
	})
}

// ListControllers obtains a list of ReplicationControllers.
func (r *Registry) ListControllers() (*api.ReplicationControllerList, error) {
	controllers := &api.ReplicationControllerList{}
	err := r.ExtractList("/registry/controllers", &controllers.Items, &controllers.ResourceVersion)
	return controllers, err
}

// WatchControllers begins watching for new, changed, or deleted controllers.
func (r *Registry) WatchControllers(resourceVersion uint64) (watch.Interface, error) {
	return r.WatchList("/registry/controllers", resourceVersion, tools.Everything)
}

func makeControllerKey(id string) string {
	return "/registry/controllers/" + id
}

// GetController gets a specific ReplicationController specified by its ID.
func (r *Registry) GetController(controllerID string) (*api.ReplicationController, error) {
	var controller api.ReplicationController
	key := makeControllerKey(controllerID)
	err := r.ExtractObj(key, &controller, false)
	if tools.IsEtcdNotFound(err) {
		return nil, apiserver.NewNotFoundErr("replicationController", controllerID)
	}
	if err != nil {
		return nil, err
	}
	return &controller, nil
}

// CreateController creates a new ReplicationController.
func (r *Registry) CreateController(controller api.ReplicationController) error {
	err := r.CreateObj(makeControllerKey(controller.ID), controller)
	if tools.IsEtcdNodeExist(err) {
		return apiserver.NewAlreadyExistsErr("replicationController", controller.ID)
	}
	return err
}

// UpdateController replaces an existing ReplicationController.
func (r *Registry) UpdateController(controller api.ReplicationController) error {
	return r.SetObj(makeControllerKey(controller.ID), controller)
}

// DeleteController deletes a ReplicationController specified by its ID.
func (r *Registry) DeleteController(controllerID string) error {
	key := makeControllerKey(controllerID)
	err := r.Delete(key, false)
	if tools.IsEtcdNotFound(err) {
		return apiserver.NewNotFoundErr("replicationController", controllerID)
	}
	return err
}

func makeServiceKey(name string) string {
	return "/registry/services/specs/" + name
}

// ListServices obtains a list of Services.
func (r *Registry) ListServices() (*api.ServiceList, error) {
	list := &api.ServiceList{}
	err := r.ExtractList("/registry/services/specs", &list.Items, &list.ResourceVersion)
	return list, err
}

// CreateService creates a new Service.
func (r *Registry) CreateService(svc api.Service) error {
	err := r.CreateObj(makeServiceKey(svc.ID), svc)
	if tools.IsEtcdNodeExist(err) {
		return apiserver.NewAlreadyExistsErr("service", svc.ID)
	}
	return err
}

// GetService obtains a Service specified by its name.
func (r *Registry) GetService(name string) (*api.Service, error) {
	key := makeServiceKey(name)
	var svc api.Service
	err := r.ExtractObj(key, &svc, false)
	if tools.IsEtcdNotFound(err) {
		return nil, apiserver.NewNotFoundErr("service", name)
	}
	if err != nil {
		return nil, err
	}
	return &svc, nil
}

// GetEndpoints obtains the endpoints for the service identified by 'name'.
func (r *Registry) GetEndpoints(name string) (*api.Endpoints, error) {
	key := makeServiceEndpointsKey(name)
	var endpoints api.Endpoints
	err := r.ExtractObj(key, &endpoints, false)
	if tools.IsEtcdNotFound(err) {
		return nil, apiserver.NewNotFoundErr("endpoints", name)
	}
	if err != nil {
		return nil, err
	}
	return &endpoints, nil
}

func makeServiceEndpointsKey(name string) string {
	return "/registry/services/endpoints/" + name
}

// DeleteService deletes a Service specified by its name.
func (r *Registry) DeleteService(name string) error {
	key := makeServiceKey(name)
	err := r.Delete(key, true)
	if tools.IsEtcdNotFound(err) {
		return apiserver.NewNotFoundErr("service", name)
	}
	if err != nil {
		return err
	}
	key = makeServiceEndpointsKey(name)
	err = r.Delete(key, true)
	if !tools.IsEtcdNotFound(err) {
		return err
	}
	return nil
}

// UpdateService replaces an existing Service.
func (r *Registry) UpdateService(svc api.Service) error {
	return r.SetObj(makeServiceKey(svc.ID), svc)
}

// WatchServices begins watching for new, changed, or deleted service configurations.
func (r *Registry) WatchServices(label, field labels.Selector, resourceVersion uint64) (watch.Interface, error) {
	if !label.Empty() {
		return nil, fmt.Errorf("label selectors are not supported on services")
	}
	if value, found := field.RequiresExactMatch("ID"); found {
		return r.Watch(makeServiceKey(value), resourceVersion)
	}
	if field.Empty() {
		return r.WatchList("/registry/services/specs", resourceVersion, tools.Everything)
	}
	return nil, fmt.Errorf("only the 'ID' and default (everything) field selectors are supported")
}

// UpdateEndpoints update Endpoints of a Service.
func (r *Registry) UpdateEndpoints(e api.Endpoints) error {
	// TODO: this is a really bad misuse of AtomicUpdate, need to compute a diff inside the loop.
	return r.AtomicUpdate(makeServiceEndpointsKey(e.ID), &api.Endpoints{},
		func(input interface{}) (interface{}, error) {
			// TODO: racy - label query is returning different results for two simultaneous updaters
			return e, nil
		})
}

// WatchEndpoints begins watching for new, changed, or deleted endpoint configurations.
func (r *Registry) WatchEndpoints(label, field labels.Selector, resourceVersion uint64) (watch.Interface, error) {
	if !label.Empty() {
		return nil, fmt.Errorf("label selectors are not supported on endpoints")
	}
	if value, found := field.RequiresExactMatch("ID"); found {
		return r.Watch(makeServiceEndpointsKey(value), resourceVersion)
	}
	if field.Empty() {
		return r.WatchList("/registry/services/endpoints", resourceVersion, tools.Everything)
	}
	return nil, fmt.Errorf("only the 'ID' and default (everything) field selectors are supported")
}
