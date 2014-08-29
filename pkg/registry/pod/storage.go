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

package pod

import (
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/apiserver"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/cloudprovider"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"

	"code.google.com/p/go-uuid/uuid"
	"github.com/golang/glog"
)

// RegistryStorage implements the RESTStorage interface in terms of a PodRegistry
type RegistryStorage struct {
	cloudProvider cloudprovider.Interface
	mu            sync.Mutex
	podCache      client.PodInfoGetter
	podInfoGetter client.PodInfoGetter
	podPollPeriod time.Duration
	registry      Registry
}

type RegistryStorageConfig struct {
	CloudProvider cloudprovider.Interface
	PodCache      client.PodInfoGetter
	PodInfoGetter client.PodInfoGetter
	Registry      Registry
}

// NewRegistryStorage returns a new RegistryStorage.
func NewRegistryStorage(config *RegistryStorageConfig) apiserver.RESTStorage {
	return &RegistryStorage{
		cloudProvider: config.CloudProvider,
		podCache:      config.PodCache,
		podInfoGetter: config.PodInfoGetter,
		podPollPeriod: time.Second * 10,
		registry:      config.Registry,
	}
}

func (rs *RegistryStorage) Create(obj interface{}) (<-chan interface{}, error) {
	pod := obj.(*api.Pod)
	if len(pod.ID) == 0 {
		pod.ID = uuid.NewUUID().String()
	}
	pod.DesiredState.Manifest.ID = pod.ID
	if errs := api.ValidatePod(pod); len(errs) > 0 {
		return nil, apiserver.NewInvalidErr("pod", pod.ID, errs)
	}

	pod.CreationTimestamp = util.Now()

	return apiserver.MakeAsync(func() (interface{}, error) {
		if err := rs.registry.CreatePod(*pod); err != nil {
			return nil, err
		}
		return rs.registry.GetPod(pod.ID)
	}), nil
}

func (rs *RegistryStorage) Delete(id string) (<-chan interface{}, error) {
	return apiserver.MakeAsync(func() (interface{}, error) {
		return &api.Status{Status: api.StatusSuccess}, rs.registry.DeletePod(id)
	}), nil
}

func (rs *RegistryStorage) Get(id string) (interface{}, error) {
	pod, err := rs.registry.GetPod(id)
	if err != nil {
		return pod, err
	}
	if pod == nil {
		return pod, nil
	}
	if rs.podCache != nil || rs.podInfoGetter != nil {
		rs.fillPodInfo(pod)
		pod.CurrentState.Status = getPodStatus(pod)
	}
	pod.CurrentState.HostIP = getInstanceIP(rs.cloudProvider, pod.CurrentState.Host)
	return pod, err
}

func (rs *RegistryStorage) List(selector labels.Selector) (interface{}, error) {
	pods, err := rs.registry.ListPods(selector)
	if err == nil {
		for i := range pods.Items {
			rs.fillPodInfo(&pods.Items[i])
		}
	}
	return pods, err
}

// Watch begins watching for new, changed, or deleted pods.
func (rs *RegistryStorage) Watch(label, field labels.Selector, resourceVersion uint64) (watch.Interface, error) {
	return rs.registry.WatchPods(resourceVersion, func(pod *api.Pod) bool {
		fields := labels.Set{
			"ID": pod.ID,
			"DesiredState.Status": string(pod.DesiredState.Status),
			"DesiredState.Host":   pod.DesiredState.Host,
		}
		return label.Matches(labels.Set(pod.Labels)) && field.Matches(fields)
	})
}

func (rs RegistryStorage) New() interface{} {
	return &api.Pod{}
}

func (rs *RegistryStorage) Update(obj interface{}) (<-chan interface{}, error) {
	pod := obj.(*api.Pod)
	if errs := api.ValidatePod(pod); len(errs) > 0 {
		return nil, apiserver.NewInvalidErr("pod", pod.ID, errs)
	}
	return apiserver.MakeAsync(func() (interface{}, error) {
		if err := rs.registry.UpdatePod(*pod); err != nil {
			return nil, err
		}
		return rs.registry.GetPod(pod.ID)
	}), nil
}

func (rs *RegistryStorage) fillPodInfo(pod *api.Pod) {
	pod.CurrentState.Host = pod.DesiredState.Host
	if pod.CurrentState.Host == "" {
		return
	}
	// Get cached info for the list currently.
	// TODO: Optionally use fresh info
	if rs.podCache != nil {
		info, err := rs.podCache.GetPodInfo(pod.CurrentState.Host, pod.ID)
		if err != nil {
			if err != client.ErrPodInfoNotAvailable {
				glog.Errorf("Error getting container info from cache: %#v", err)
			}
			if rs.podInfoGetter != nil {
				info, err = rs.podInfoGetter.GetPodInfo(pod.CurrentState.Host, pod.ID)
			}
			if err != nil {
				if err != client.ErrPodInfoNotAvailable {
					glog.Errorf("Error getting fresh container info: %#v", err)
				}
				return
			}
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
			glog.Warningf("Couldn't find network container for %s in %v", pod.ID, info)
		}
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

func getPodStatus(pod *api.Pod) api.PodStatus {
	if pod.CurrentState.Info == nil || pod.CurrentState.Host == "" {
		return api.PodWaiting
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
		return api.PodTerminated
	case running == 0 && stopped == 0 && unknown > 0:
		return api.PodWaiting
	default:
		return api.PodWaiting
	}
}

func (rs *RegistryStorage) waitForPodRunning(pod api.Pod) (interface{}, error) {
	for {
		podObj, err := rs.Get(pod.ID)
		if err != nil || podObj == nil {
			return nil, err
		}
		podPtr, ok := podObj.(*api.Pod)
		if !ok {
			// This should really never happen.
			return nil, fmt.Errorf("Error %#v is not an api.Pod!", podObj)
		}
		switch podPtr.CurrentState.Status {
		case api.PodRunning, api.PodTerminated:
			return pod, nil
		default:
			time.Sleep(rs.podPollPeriod)
		}
	}
	return pod, nil
}
