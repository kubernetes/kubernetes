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
	"encoding/json"
	"fmt"
	"log"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

// TODO: Need to add a reconciler loop that makes sure that things in pods are reflected into
//       kubelet (and vice versa)

// EtcdRegistry is an implementation of both ControllerRegistry and PodRegistry which is backed with etcd.
type EtcdRegistry struct {
	etcdClient      util.EtcdClient
	machines        MinionRegistry
	manifestFactory ManifestFactory
}

// MakeEtcdRegistry creates an etcd registry.
// 'client' is the connection to etcd
// 'machines' is the list of machines
// 'scheduler' is the scheduling algorithm to use.
func MakeEtcdRegistry(client util.EtcdClient, machines MinionRegistry) *EtcdRegistry {
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

func (registry *EtcdRegistry) helper() *util.EtcdHelper {
	return &util.EtcdHelper{registry.etcdClient}
}

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

func (registry *EtcdRegistry) GetPod(podID string) (*api.Pod, error) {
	pod, _, err := registry.findPod(podID)
	return &pod, err
}

func makeContainerKey(machine string) string {
	return "/registry/hosts/" + machine + "/kubelet"
}

func (registry *EtcdRegistry) loadManifests(machine string) (manifests []api.ContainerManifest, err error) {
	err = registry.helper().ExtractObj(makeContainerKey(machine), &manifests, true)
	return
}

func (registry *EtcdRegistry) updateManifests(machine string, manifests []api.ContainerManifest) error {
	return registry.helper().SetObj(makeContainerKey(machine), manifests)
}

func (registry *EtcdRegistry) CreatePod(machineIn string, pod api.Pod) error {
	podOut, machine, err := registry.findPod(pod.ID)
	if err == nil {
		return fmt.Errorf("a pod named %s already exists on %s (%#v)", pod.ID, machine, podOut)
	}
	return registry.runPod(pod, machineIn)
}

func (registry *EtcdRegistry) runPod(pod api.Pod, machine string) error {
	manifests, err := registry.loadManifests(machine)
	if err != nil {
		return err
	}

	key := makePodKey(machine, pod.ID)
	data, err := json.Marshal(pod)
	if err != nil {
		return err
	}
	_, err = registry.etcdClient.Create(key, string(data), 0)

	manifest, err := registry.manifestFactory.MakeManifest(machine, pod)
	if err != nil {
		return err
	}
	manifests = append(manifests, manifest)
	return registry.updateManifests(machine, manifests)
}

func (registry *EtcdRegistry) UpdatePod(pod api.Pod) error {
	return fmt.Errorf("unimplemented!")
}

func (registry *EtcdRegistry) DeletePod(podID string) error {
	_, machine, err := registry.findPod(podID)
	if err != nil {
		return err
	}
	return registry.deletePodFromMachine(machine, podID)
}

func (registry *EtcdRegistry) deletePodFromMachine(machine, podID string) error {
	manifests, err := registry.loadManifests(machine)
	if err != nil {
		return err
	}
	newManifests := make([]api.ContainerManifest, 0)
	found := false
	for _, manifest := range manifests {
		if manifest.Id != podID {
			newManifests = append(newManifests, manifest)
		} else {
			found = true
		}
	}
	if !found {
		// This really shouldn't happen, it indicates something is broken, and likely
		// there is a lost pod somewhere.
		// However it is "deleted" so log it and move on
		log.Printf("Couldn't find: %s in %#v", podID, manifests)
	}
	if err = registry.updateManifests(machine, newManifests); err != nil {
		return err
	}
	key := makePodKey(machine, podID)
	_, err = registry.etcdClient.Delete(key, true)
	return err
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
	return api.Pod{}, "", fmt.Errorf("pod not found %s", podID)
}

func (registry *EtcdRegistry) ListControllers() ([]api.ReplicationController, error) {
	var controllers []api.ReplicationController
	err := registry.helper().ExtractList("/registry/controllers", &controllers)
	return controllers, err
}

func makeControllerKey(id string) string {
	return "/registry/controllers/" + id
}

func (registry *EtcdRegistry) GetController(controllerID string) (*api.ReplicationController, error) {
	var controller api.ReplicationController
	key := makeControllerKey(controllerID)
	err := registry.helper().ExtractObj(key, &controller, false)
	if err != nil {
		return nil, err
	}
	return &controller, nil
}

func (registry *EtcdRegistry) CreateController(controller api.ReplicationController) error {
	// TODO : check for existence here and error.
	return registry.UpdateController(controller)
}

func (registry *EtcdRegistry) UpdateController(controller api.ReplicationController) error {
	return registry.helper().SetObj(makeControllerKey(controller.ID), controller)
}

func (registry *EtcdRegistry) DeleteController(controllerID string) error {
	key := makeControllerKey(controllerID)
	_, err := registry.etcdClient.Delete(key, false)
	return err
}

func makeServiceKey(name string) string {
	return "/registry/services/specs/" + name
}

func (registry *EtcdRegistry) ListServices() (api.ServiceList, error) {
	var list api.ServiceList
	err := registry.helper().ExtractList("/registry/services/specs", &list.Items)
	return list, err
}

func (registry *EtcdRegistry) CreateService(svc api.Service) error {
	return registry.helper().SetObj(makeServiceKey(svc.ID), svc)
}

func (registry *EtcdRegistry) GetService(name string) (*api.Service, error) {
	key := makeServiceKey(name)
	var svc api.Service
	err := registry.helper().ExtractObj(key, &svc, false)
	if err != nil {
		return nil, err
	}
	return &svc, nil
}

func (registry *EtcdRegistry) DeleteService(name string) error {
	key := makeServiceKey(name)
	_, err := registry.etcdClient.Delete(key, true)
	if err != nil {
		return err
	}
	key = "/registry/services/endpoints/" + name
	_, err = registry.etcdClient.Delete(key, true)
	return err
}

func (registry *EtcdRegistry) UpdateService(svc api.Service) error {
	return registry.CreateService(svc)
}

func (registry *EtcdRegistry) UpdateEndpoints(e api.Endpoints) error {
	return registry.helper().SetObj("/registry/services/endpoints/"+e.Name, e)
}
