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
	"net/url"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/apiserver"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
)

// PodRegistryStorage implements the RESTStorage interface in terms of a PodRegistry
type PodRegistryStorage struct {
	registry      PodRegistry
	containerInfo client.ContainerInfo
	scheduler     Scheduler
}

func MakePodRegistryStorage(registry PodRegistry, containerInfo client.ContainerInfo, scheduler Scheduler) apiserver.RESTStorage {
	return &PodRegistryStorage{
		registry:      registry,
		containerInfo: containerInfo,
		scheduler:     scheduler,
	}
}

// LabelMatch tests to see if a Pod's labels map contains 'key' mapping to 'value'
func LabelMatch(pod api.Pod, queryKey, queryValue string) bool {
	for key, value := range pod.Labels {
		if queryKey == key && queryValue == value {
			return true
		}
	}
	return false
}

// LabelMatch tests to see if a Pod's labels map contains all key/value pairs in 'labelQuery'
func LabelsMatch(pod api.Pod, labelQuery *map[string]string) bool {
	if labelQuery == nil {
		return true
	}
	for key, value := range *labelQuery {
		if !LabelMatch(pod, key, value) {
			return false
		}
	}
	return true
}

func (storage *PodRegistryStorage) List(url *url.URL) (interface{}, error) {
	var result api.PodList
	var query *map[string]string
	if url != nil {
		queryMap := client.DecodeLabelQuery(url.Query().Get("labels"))
		query = &queryMap
	}
	pods, err := storage.registry.ListPods(query)
	if err == nil {
		result = api.PodList{
			Items: pods,
		}
	}
	result.Kind = "cluster#podList"
	return result, err
}

func makePodStatus(info interface{}) string {
	if state, ok := info.(map[string]interface{})["State"]; ok {
		if running, ok := state.(map[string]interface{})["Running"]; ok {
			if running.(bool) {
				return "Running"
			} else {
				return "Stopped"
			}
		}
	}
	return "Pending"
}

func (storage *PodRegistryStorage) Get(id string) (interface{}, error) {
	pod, err := storage.registry.GetPod(id)
	if err != nil {
		return pod, err
	}
	info, err := storage.containerInfo.GetContainerInfo(pod.CurrentState.Host, id)
	if err != nil {
		return pod, err
	}
	pod.CurrentState.Info = info
	pod.CurrentState.Status = makePodStatus(info)
	pod.Kind = "cluster#pod"
	return pod, err
}

func (storage *PodRegistryStorage) Delete(id string) error {
	return storage.registry.DeletePod(id)
}

func (storage *PodRegistryStorage) Extract(body string) (interface{}, error) {
	pod := api.Pod{}
	err := json.Unmarshal([]byte(body), &pod)
	pod.Kind = "cluster#pod"
	return pod, err
}

func (storage *PodRegistryStorage) Create(pod interface{}) error {
	podObj := pod.(api.Pod)
	if len(podObj.ID) == 0 {
		return fmt.Errorf("id is unspecified: %#v", pod)
	}
	machine, err := storage.scheduler.Schedule(podObj)
	if err != nil {
		return err
	}
	return storage.registry.CreatePod(machine, podObj)
}

func (storage *PodRegistryStorage) Update(pod interface{}) error {
	return storage.registry.UpdatePod(pod.(api.Pod))
}
