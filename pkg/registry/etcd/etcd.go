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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/minion"
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
func NewRegistry(client tools.EtcdClient, machines minion.Registry) *Registry {
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
func (r *Registry) ListPods(selector labels.Selector) ([]api.Pod, error) {
	allPods := []api.Pod{}
	filteredPods := []api.Pod{}
	if err := r.ExtractList("/registry/pods", &allPods); err != nil {
		return nil, err
	}
	for _, pod := range allPods {
		if selector.Matches(labels.Set(pod.Labels)) {
			// TODO: Currently nothing sets CurrentState.Host. We need a feedback loop that sets
			// the CurrentState.Host and Status fields. Here we pretend that reality perfectly
			// matches our desires.
			pod.CurrentState.Host = pod.DesiredState.Host
			filteredPods = append(filteredPods, pod)
		}
	}
	return filteredPods, nil
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

// CreatePod creates a pod based on a specification, schedule it onto a specific machine.
func (r *Registry) CreatePod(machine string, pod api.Pod) error {
	// Set current status to "Waiting".
	pod.CurrentState.Status = api.PodWaiting
	pod.CurrentState.Host = ""
	// DesiredState.Host == "" is a signal to the scheduler that this pod needs scheduling.
	pod.DesiredState.Status = api.PodRunning
	pod.DesiredState.Host = ""
	if err := r.CreateObj(makePodKey(pod.ID), &pod); err != nil {
		return err
	}
	// TODO: Until scheduler separation is completed, just assign here.
	return r.AssignPod(pod.ID, machine)
}

// AssignPod assigns the given pod to the given machine.
// TODO: hook this up via apiserver, not by calling it from CreatePod().
func (r *Registry) AssignPod(podID string, machine string) error {
	podKey := makePodKey(podID)
	var finalPod *api.Pod
	err := r.AtomicUpdate(podKey, &api.Pod{}, func(obj interface{}) (interface{}, error) {
		pod, ok := obj.(*api.Pod)
		if !ok {
			return nil, fmt.Errorf("unexpected object: %#v", obj)
		}
		pod.DesiredState.Host = machine
		finalPod = pod
		return pod, nil
	})
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
		return manifests, nil
	})
	if err != nil {
		// Don't strand stuff. This is a terrible hack that won't be needed
		// when the above TODO is fixed.
		err2 := r.Delete(podKey, false)
		if err2 != nil {
			glog.Errorf("Probably stranding a pod, couldn't delete %v: %#v", podKey, err2)
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
func (r *Registry) ListControllers() ([]api.ReplicationController, error) {
	var controllers []api.ReplicationController
	err := r.ExtractList("/registry/controllers", &controllers)
	return controllers, err
}

// WatchControllers begins watching for new, changed, or deleted controllers.
func (r *Registry) WatchControllers(label, field labels.Selector, resourceVersion uint64) (watch.Interface, error) {
	if !field.Empty() {
		return nil, fmt.Errorf("no field selector implemented for controllers")
	}
	return r.WatchList("/registry/controllers", resourceVersion, func(obj interface{}) bool {
		return label.Matches(labels.Set(obj.(*api.ReplicationController).Labels))
	})
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
func (r *Registry) ListServices() (api.ServiceList, error) {
	var list api.ServiceList
	err := r.ExtractList("/registry/services/specs", &list.Items)
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

// UpdateEndpoints update Endpoints of a Service.
func (r *Registry) UpdateEndpoints(e api.Endpoints) error {
	return r.AtomicUpdate(makeServiceEndpointsKey(e.ID), &api.Endpoints{},
		func(interface{}) (interface{}, error) {
			return e, nil
		})
}
