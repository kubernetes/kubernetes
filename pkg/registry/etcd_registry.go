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

package registry

import (
	"fmt"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/apiserver"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/tools"
	"github.com/golang/glog"
)

// TODO: Need to add a reconciler loop that makes sure that things in pods are reflected into
//       kubelet (and vice versa)

// EtcdRegistry implements PodRegistry, ControllerRegistry and ServiceRegistry with backed by etcd.
type EtcdRegistry struct {
	etcdClient      tools.EtcdClient
	machines        MinionRegistry
	manifestFactory ManifestFactory
}

// MakeEtcdRegistry creates an etcd registry.
// 'client' is the connection to etcd
// 'machines' is the list of machines
// 'scheduler' is the scheduling algorithm to use.
func MakeEtcdRegistry(client tools.EtcdClient, machines MinionRegistry) *EtcdRegistry {
	registry := &EtcdRegistry{
		etcdClient: client,
		machines:   machines,
	}
	registry.manifestFactory = &BasicManifestFactory{
		serviceRegistry: registry,
	}
	return registry
}

func makePodKey(machine, podID string) string {
	return "/registry/hosts/" + machine + "/pods/" + podID
}

func (registry *EtcdRegistry) helper() *tools.EtcdHelper {
	return &tools.EtcdHelper{registry.etcdClient}
}

// ListPods obtains a list of pods that match selector.
func (registry *EtcdRegistry) ListPods(selector labels.Selector) ([]api.Pod, error) {
	pods := []api.Pod{}
	machines, err := registry.machines.List()
	if err != nil {
		return nil, err
	}
	for _, machine := range machines {
		var machinePods []api.Pod
		err := registry.helper().ExtractList("/registry/hosts/"+machine+"/pods", &machinePods)
		if err != nil {
			return pods, err
		}
		for _, pod := range machinePods {
			if selector.Matches(labels.Set(pod.Labels)) {
				pod.CurrentState.Host = machine
				pods = append(pods, pod)
			}
		}
	}
	return pods, nil
}

// GetPod gets a specific pod specified by its ID.
func (registry *EtcdRegistry) GetPod(podID string) (*api.Pod, error) {
	pod, _, err := registry.findPod(podID)
	return &pod, err
}

func makeContainerKey(machine string) string {
	return "/registry/hosts/" + machine + "/kubelet"
}

// CreatePod creates a pod based on a specification, schedule it onto a specific machine.
func (registry *EtcdRegistry) CreatePod(machineIn string, pod api.Pod) error {
	podOut, machine, err := registry.findPod(pod.ID)
	if err == nil {
		// TODO: this error message looks racy.
		return fmt.Errorf("a pod named %s already exists on %s (%#v)", pod.ID, machine, podOut)
	}
	return registry.runPod(pod, machineIn)
}

func (registry *EtcdRegistry) runPod(pod api.Pod, machine string) error {
	podKey := makePodKey(machine, pod.ID)
	err := registry.helper().SetObj(podKey, pod)

	manifest, err := registry.manifestFactory.MakeManifest(machine, pod)
	if err != nil {
		return err
	}

	contKey := makeContainerKey(machine)
	err = registry.helper().AtomicUpdate(contKey, &[]api.ContainerManifest{}, func(in interface{}) (interface{}, error) {
		manifests := *in.(*[]api.ContainerManifest)
		return append(manifests, manifest), nil
	})
	if err != nil {
		// Don't strand stuff.
		_, err2 := registry.etcdClient.Delete(podKey, false)
		if err2 != nil {
			glog.Errorf("Probably stranding a pod, couldn't delete %v: %#v", podKey, err2)
		}
	}
	return err
}

func (registry *EtcdRegistry) UpdatePod(pod api.Pod) error {
	return fmt.Errorf("unimplemented!")
}

// DeletePod deletes an existing pod specified by its ID.
func (registry *EtcdRegistry) DeletePod(podID string) error {
	_, machine, err := registry.findPod(podID)
	if err != nil {
		return err
	}
	return registry.deletePodFromMachine(machine, podID)
}

func (registry *EtcdRegistry) deletePodFromMachine(machine, podID string) error {
	// First delete the pod, so a scheduler doesn't notice it getting removed from the
	// machine and attempt to put it somewhere.
	podKey := makePodKey(machine, podID)
	_, err := registry.etcdClient.Delete(podKey, true)
	if tools.IsEtcdNotFound(err) {
		return apiserver.NewNotFoundErr("pod", podID)
	}
	if err != nil {
		return err
	}

	// Next, remove the pod from the machine atomically.
	contKey := makeContainerKey(machine)
	return registry.helper().AtomicUpdate(contKey, &[]api.ContainerManifest{}, func(in interface{}) (interface{}, error) {
		manifests := *in.(*[]api.ContainerManifest)
		newManifests := make([]api.ContainerManifest, 0, len(manifests))
		found := false
		for _, manifest := range manifests {
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
		return newManifests, nil
	})
}

func (registry *EtcdRegistry) getPodForMachine(machine, podID string) (pod api.Pod, err error) {
	key := makePodKey(machine, podID)
	err = registry.helper().ExtractObj(key, &pod, false)
	if err != nil {
		return
	}
	pod.CurrentState.Host = machine
	return
}

func (registry *EtcdRegistry) findPod(podID string) (api.Pod, string, error) {
	machines, err := registry.machines.List()
	if err != nil {
		return api.Pod{}, "", err
	}
	for _, machine := range machines {
		pod, err := registry.getPodForMachine(machine, podID)
		if err == nil {
			return pod, machine, nil
		}
	}
	return api.Pod{}, "", apiserver.NewNotFoundErr("pod", podID)
}

// ListControllers obtains a list of ReplicationControllers.
func (registry *EtcdRegistry) ListControllers() ([]api.ReplicationController, error) {
	var controllers []api.ReplicationController
	err := registry.helper().ExtractList("/registry/controllers", &controllers)
	return controllers, err
}

func makeControllerKey(id string) string {
	return "/registry/controllers/" + id
}

// GetController gets a specific ReplicationController specified by its ID.
func (registry *EtcdRegistry) GetController(controllerID string) (*api.ReplicationController, error) {
	var controller api.ReplicationController
	key := makeControllerKey(controllerID)
	err := registry.helper().ExtractObj(key, &controller, false)
	if tools.IsEtcdNotFound(err) {
		return nil, apiserver.NewNotFoundErr("replicationController", controllerID)
	}
	if err != nil {
		return nil, err
	}
	return &controller, nil
}

// CreateController creates a new ReplicationController.
func (registry *EtcdRegistry) CreateController(controller api.ReplicationController) error {
	// TODO : check for existence here and error.
	return registry.UpdateController(controller)
}

// UpdateController replaces an existing ReplicationController.
func (registry *EtcdRegistry) UpdateController(controller api.ReplicationController) error {
	return registry.helper().SetObj(makeControllerKey(controller.ID), controller)
}

// DeleteController deletes a ReplicationController specified by its ID.
func (registry *EtcdRegistry) DeleteController(controllerID string) error {
	key := makeControllerKey(controllerID)
	_, err := registry.etcdClient.Delete(key, false)
	if tools.IsEtcdNotFound(err) {
		return apiserver.NewNotFoundErr("replicationController", controllerID)
	}
	return err
}

func makeServiceKey(name string) string {
	return "/registry/services/specs/" + name
}

// ListServices obtains a list of Services.
func (registry *EtcdRegistry) ListServices() (api.ServiceList, error) {
	var list api.ServiceList
	err := registry.helper().ExtractList("/registry/services/specs", &list.Items)
	return list, err
}

// CreateService creates a new Service.
func (registry *EtcdRegistry) CreateService(svc api.Service) error {
	return registry.helper().SetObj(makeServiceKey(svc.ID), svc)
}

// GetService obtains a Service specified by its name.
func (registry *EtcdRegistry) GetService(name string) (*api.Service, error) {
	key := makeServiceKey(name)
	var svc api.Service
	err := registry.helper().ExtractObj(key, &svc, false)
	if tools.IsEtcdNotFound(err) {
		return nil, apiserver.NewNotFoundErr("service", name)
	}
	if err != nil {
		return nil, err
	}
	return &svc, nil
}

// DeleteService deletes a Service specified by its name.
func (registry *EtcdRegistry) DeleteService(name string) error {
	key := makeServiceKey(name)
	_, err := registry.etcdClient.Delete(key, true)
	if tools.IsEtcdNotFound(err) {
		return apiserver.NewNotFoundErr("service", name)
	}
	if err != nil {
		return err
	}
	key = "/registry/services/endpoints/" + name
	_, err = registry.etcdClient.Delete(key, true)
	return err
}

// UpdateService replaces an existing Service.
func (registry *EtcdRegistry) UpdateService(svc api.Service) error {
	// TODO : check for existence here and error.
	return registry.CreateService(svc)
}

// UpdateEndpoints update Endpoints of a Service.
func (registry *EtcdRegistry) UpdateEndpoints(e api.Endpoints) error {
	return registry.helper().SetObj("/registry/services/endpoints/"+e.Name, e)
}
