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
	"strings"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/apiserver"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/cloudprovider"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/scheduler"
	"github.com/golang/glog"
)

// PodRegistryStorage implements the RESTStorage interface in terms of a PodRegistry
type PodRegistryStorage struct {
	registry      PodRegistry
	containerInfo client.ContainerInfo
	podCache      client.ContainerInfo
	scheduler     scheduler.Scheduler
	minionLister  scheduler.MinionLister
	cloud         cloudprovider.Interface
}

// MakePodRegistryStorage makes a RESTStorage object for a pod registry.
// Parameters:
//   registry:      The pod registry
//   containerInfo: Source of fresh container info
//   scheduler:     The scheduler for assigning pods to machines
//   minionLister:  Object which can list available minions for the scheduler
//   cloud:         Interface to a cloud provider (may be null)
//   podCache:      Source of cached container info
func MakePodRegistryStorage(registry PodRegistry,
	containerInfo client.ContainerInfo,
	scheduler scheduler.Scheduler,
	minionLister scheduler.MinionLister,
	cloud cloudprovider.Interface,
	podCache client.ContainerInfo) apiserver.RESTStorage {
	return &PodRegistryStorage{
		registry:      registry,
		containerInfo: containerInfo,
		scheduler:     scheduler,
		minionLister:  minionLister,
		cloud:         cloud,
		podCache:      podCache,
	}
}

func (storage *PodRegistryStorage) List(selector labels.Selector) (interface{}, error) {
	var result api.PodList
	pods, err := storage.registry.ListPods(selector)
	if err == nil {
		result.Items = pods
		// Get cached info for the list currently.
		// TODO: Optionally use fresh info
		if storage.podCache != nil {
			for ix, pod := range pods {
				info, err := storage.podCache.GetContainerInfo(pod.CurrentState.Host, pod.ID)
				if err != nil {
					glog.Errorf("Error getting container info: %#v", err)
					continue
				}
				result.Items[ix].CurrentState.Info = info
			}
		}
	}

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

func getInstanceIP(cloud cloudprovider.Interface, host string) string {
	if cloud == nil {
		return ""
	}
	instances, ok := cloud.Instances()
	if instances == nil || !ok {
		return ""
	}
	ix := strings.Index(host, ".")
	if ix != -1 {
		host = host[:ix]
	}
	addr, err := instances.IPAddress(host)
	if err != nil {
		glog.Errorf("Error getting instance IP: %#v", err)
		return ""
	}
	return addr.String()
}

func (storage *PodRegistryStorage) Get(id string) (interface{}, error) {
	pod, err := storage.registry.GetPod(id)
	if err != nil {
		return pod, err
	}
	if pod == nil {
		return pod, nil
	}
	if storage.containerInfo != nil {
		info, err := storage.containerInfo.GetContainerInfo(pod.CurrentState.Host, id)
		if err != nil {
			return pod, err
		}
		pod.CurrentState.Info = info
		pod.CurrentState.Status = makePodStatus(info)
	}
	pod.CurrentState.HostIP = getInstanceIP(storage.cloud, pod.CurrentState.Host)

	return pod, err
}

func (storage *PodRegistryStorage) Delete(id string) (<-chan interface{}, error) {
	return apiserver.MakeAsync(func() (interface{}, error) {
		return api.Status{Status: api.StatusSuccess}, storage.registry.DeletePod(id)
	}), nil
}

func (storage *PodRegistryStorage) Extract(body []byte) (interface{}, error) {
	pod := api.Pod{}
	err := api.DecodeInto(body, &pod)
	return pod, err
}

func (storage *PodRegistryStorage) Create(obj interface{}) (<-chan interface{}, error) {
	pod := obj.(api.Pod)
	if len(pod.ID) == 0 {
		return nil, fmt.Errorf("id is unspecified: %#v", pod)
	}

	return apiserver.MakeAsync(func() (interface{}, error) {
		// TODO(lavalamp): Separate scheduler more cleanly.
		machine, err := storage.scheduler.Schedule(pod, storage.minionLister)
		if err != nil {
			return nil, err
		}
		err = storage.registry.CreatePod(machine, pod)
		if err != nil {
			return nil, err
		}
		return storage.registry.GetPod(pod.ID)
	}), nil
}

func (storage *PodRegistryStorage) Update(obj interface{}) (<-chan interface{}, error) {
	pod := obj.(api.Pod)
	if len(pod.ID) == 0 {
		return nil, fmt.Errorf("id is unspecified: %#v", pod)
	}

	return apiserver.MakeAsync(func() (interface{}, error) {
		err := storage.registry.UpdatePod(pod)
		if err != nil {
			return nil, err
		}
		return storage.registry.GetPod(pod.ID)
	}), nil
}
