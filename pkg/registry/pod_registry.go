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
	"time"

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
	podInfoGetter client.PodInfoGetter
	podCache      client.PodInfoGetter
	scheduler     scheduler.Scheduler
	minionLister  scheduler.MinionLister
	cloud         cloudprovider.Interface
	podPollPeriod time.Duration
}

// MakePodRegistryStorage makes a RESTStorage object for a pod registry.
// Parameters:
//   registry:      The pod registry
//   podInfoGetter: Source of fresh container info
//   scheduler:     The scheduler for assigning pods to machines
//   minionLister:  Object which can list available minions for the scheduler
//   cloud:         Interface to a cloud provider (may be null)
//   podCache:      Source of cached container info
func MakePodRegistryStorage(registry PodRegistry,
	podInfoGetter client.PodInfoGetter,
	scheduler scheduler.Scheduler,
	minionLister scheduler.MinionLister,
	cloud cloudprovider.Interface,
	podCache client.PodInfoGetter) apiserver.RESTStorage {
	return &PodRegistryStorage{
		registry:      registry,
		podInfoGetter: podInfoGetter,
		scheduler:     scheduler,
		minionLister:  minionLister,
		cloud:         cloud,
		podCache:      podCache,
		podPollPeriod: time.Second * 10,
	}
}

func (storage *PodRegistryStorage) List(selector labels.Selector) (interface{}, error) {
	var result api.PodList
	pods, err := storage.registry.ListPods(selector)
	if err == nil {
		result.Items = pods
		for i := range result.Items {
			storage.fillPodInfo(&result.Items[i])
		}
	}

	return result, err
}

func (storage *PodRegistryStorage) fillPodInfo(pod *api.Pod) {
	// Get cached info for the list currently.
	// TODO: Optionally use fresh info
	if storage.podCache != nil {
		info, err := storage.podCache.GetPodInfo(pod.CurrentState.Host, pod.ID)
		if err != nil {
			glog.Errorf("Error getting container info: %#v", err)
			return
		}
		pod.CurrentState.Info = info
		netContainerInfo, ok := info["net"]
		if ok {
			if netContainerInfo.NetworkSettings != nil {
				pod.CurrentState.PodIP = netContainerInfo.NetworkSettings.IPAddress
			} else {
				glog.Warningf("No network settings: %#v", netContainerInfo)
			}
		} else {
			glog.Warningf("Couldn't find network container in %v", info)
		}
	}
}

func makePodStatus(pod *api.Pod) api.PodStatus {
	if pod.CurrentState.Info == nil {
		return api.PodPending
	}
	running := 0
	stopped := 0
	unknown := 0
	for _, container := range pod.DesiredState.Manifest.Containers {
		if info, ok := pod.CurrentState.Info[container.Name]; ok {
			if info.State.Running {
				running++
			} else {
				stopped++
			}
		} else {
			unknown++
		}
	}

	switch {
	case running > 0 && stopped == 0 && unknown == 0:
		return api.PodRunning
	case running == 0 && stopped > 0 && unknown == 0:
		return api.PodStopped
	case running == 0 && stopped == 0 && unknown > 0:
		return api.PodPending
	default:
		return api.PodPending
	}
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
	if storage.podCache != nil || storage.podInfoGetter != nil {
		storage.fillPodInfo(pod)
		pod.CurrentState.Status = makePodStatus(pod)
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
		return storage.waitForPodRunning(pod)
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
		return storage.waitForPodRunning(pod)
	}), nil
}

func (storage *PodRegistryStorage) waitForPodRunning(pod api.Pod) (interface{}, error) {
	for {
		podObj, err := storage.Get(pod.ID)

		if err != nil || podObj == nil {
			return nil, err
		}
		podPtr, ok := podObj.(*api.Pod)
		if !ok {
			// This should really never happen.
			return nil, fmt.Errorf("Error %#v is not an api.Pod!", podObj)
		}
		switch podPtr.CurrentState.Status {
		case api.PodRunning, api.PodStopped:
			return pod, nil
		default:
			time.Sleep(storage.podPollPeriod)
		}
	}
	return pod, nil
}
