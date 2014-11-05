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
	"sync"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/validation"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/apiserver"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/cloudprovider"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"

	"github.com/golang/glog"
)

type ipCacheEntry struct {
	ip         string
	lastUpdate time.Time
}

type ipCache map[string]ipCacheEntry

type clock interface {
	Now() time.Time
}

type realClock struct{}

func (r realClock) Now() time.Time {
	return time.Now()
}

// REST implements the RESTStorage interface in terms of a PodRegistry.
type REST struct {
	cloudProvider cloudprovider.Interface
	mu            sync.Mutex
	podCache      client.PodInfoGetter
	podInfoGetter client.PodInfoGetter
	podPollPeriod time.Duration
	registry      Registry
	minions       client.MinionInterface
	ipCache       ipCache
	clock         clock
}

type RESTConfig struct {
	CloudProvider cloudprovider.Interface
	PodCache      client.PodInfoGetter
	PodInfoGetter client.PodInfoGetter
	Registry      Registry
	Minions       client.MinionInterface
}

// NewREST returns a new REST.
func NewREST(config *RESTConfig) *REST {
	return &REST{
		cloudProvider: config.CloudProvider,
		podCache:      config.PodCache,
		podInfoGetter: config.PodInfoGetter,
		podPollPeriod: time.Second * 10,
		registry:      config.Registry,
		minions:       config.Minions,
		ipCache:       ipCache{},
		clock:         realClock{},
	}
}

func (rs *REST) Create(ctx api.Context, obj runtime.Object) (<-chan apiserver.RESTResult, error) {
	pod := obj.(*api.Pod)
	if !api.ValidNamespace(ctx, &pod.ObjectMeta) {
		return nil, errors.NewConflict("pod", pod.Namespace, fmt.Errorf("Pod.Namespace does not match the provided context"))
	}
	pod.DesiredState.Manifest.UUID = util.NewUUID().String()
	if len(pod.Name) == 0 {
		pod.Name = pod.DesiredState.Manifest.UUID
	}
	pod.DesiredState.Manifest.ID = pod.Name
	if errs := validation.ValidatePod(pod); len(errs) > 0 {
		return nil, errors.NewInvalid("pod", pod.Name, errs)
	}
	pod.CreationTimestamp = util.Now()

	return apiserver.MakeAsync(func() (runtime.Object, error) {
		if err := rs.registry.CreatePod(ctx, pod); err != nil {
			return nil, err
		}
		return rs.registry.GetPod(ctx, pod.Name)
	}), nil
}

func (rs *REST) Delete(ctx api.Context, id string) (<-chan apiserver.RESTResult, error) {
	return apiserver.MakeAsync(func() (runtime.Object, error) {
		return &api.Status{Status: api.StatusSuccess}, rs.registry.DeletePod(ctx, id)
	}), nil
}

func (rs *REST) Get(ctx api.Context, id string) (runtime.Object, error) {
	pod, err := rs.registry.GetPod(ctx, id)
	if err != nil {
		return pod, err
	}
	if pod == nil {
		return pod, nil
	}
	if rs.podCache != nil || rs.podInfoGetter != nil {
		rs.fillPodInfo(pod)
		status, err := getPodStatus(pod, rs.minions)
		if err != nil {
			return pod, err
		}
		pod.CurrentState.Status = status
	}
	if pod.CurrentState.Host != "" {
		pod.CurrentState.HostIP = rs.getInstanceIP(pod.CurrentState.Host)
	}
	return pod, err
}

func (rs *REST) podToSelectableFields(pod *api.Pod) labels.Set {
	return labels.Set{
		"name":                pod.Name,
		"DesiredState.Status": string(pod.DesiredState.Status),
		"DesiredState.Host":   pod.DesiredState.Host,
	}
}

// filterFunc returns a predicate based on label & field selectors that can be passed to registry's
// ListPods & WatchPods.
func (rs *REST) filterFunc(label, field labels.Selector) func(*api.Pod) bool {
	return func(pod *api.Pod) bool {
		fields := rs.podToSelectableFields(pod)
		return label.Matches(labels.Set(pod.Labels)) && field.Matches(fields)
	}
}

func (rs *REST) List(ctx api.Context, label, field labels.Selector) (runtime.Object, error) {
	pods, err := rs.registry.ListPodsPredicate(ctx, rs.filterFunc(label, field))
	if err == nil {
		for i := range pods.Items {
			pod := &pods.Items[i]
			rs.fillPodInfo(pod)
			status, err := getPodStatus(pod, rs.minions)
			if err != nil {
				return pod, err
			}
			pod.CurrentState.Status = status
			if pod.CurrentState.Host != "" {
				pod.CurrentState.HostIP = rs.getInstanceIP(pod.CurrentState.Host)
			}
		}
	}
	return pods, err
}

// Watch begins watching for new, changed, or deleted pods.
func (rs *REST) Watch(ctx api.Context, label, field labels.Selector, resourceVersion string) (watch.Interface, error) {
	return rs.registry.WatchPods(ctx, resourceVersion, rs.filterFunc(label, field))
}

func (*REST) New() runtime.Object {
	return &api.Pod{}
}

func (rs *REST) Update(ctx api.Context, obj runtime.Object) (<-chan apiserver.RESTResult, error) {
	pod := obj.(*api.Pod)
	if !api.ValidNamespace(ctx, &pod.ObjectMeta) {
		return nil, errors.NewConflict("pod", pod.Namespace, fmt.Errorf("Pod.Namespace does not match the provided context"))
	}
	if errs := validation.ValidatePod(pod); len(errs) > 0 {
		return nil, errors.NewInvalid("pod", pod.Name, errs)
	}
	return apiserver.MakeAsync(func() (runtime.Object, error) {
		if err := rs.registry.UpdatePod(ctx, pod); err != nil {
			return nil, err
		}
		return rs.registry.GetPod(ctx, pod.Name)
	}), nil
}

func (rs *REST) fillPodInfo(pod *api.Pod) {
	pod.CurrentState.Host = pod.DesiredState.Host
	if pod.CurrentState.Host == "" {
		return
	}
	// Get cached info for the list currently.
	// TODO: Optionally use fresh info
	if rs.podCache != nil {
		info, err := rs.podCache.GetPodInfo(pod.CurrentState.Host, pod.Namespace, pod.Name)
		if err != nil {
			if err != client.ErrPodInfoNotAvailable {
				glog.Errorf("Error getting container info from cache: %#v", err)
			}
			if rs.podInfoGetter != nil {
				info, err = rs.podInfoGetter.GetPodInfo(pod.CurrentState.Host, pod.Namespace, pod.Name)
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
			if netContainerInfo.PodIP != "" {
				pod.CurrentState.PodIP = netContainerInfo.PodIP
			} else {
				glog.Warningf("No network settings: %#v", netContainerInfo)
			}
		} else {
			glog.Warningf("Couldn't find network container for %s in %v", pod.Name, info)
		}
	}
}

func (rs *REST) getInstanceIP(host string) string {
	data, ok := rs.ipCache[host]
	now := rs.clock.Now()

	if !ok || now.Sub(data.lastUpdate) > (30*time.Second) {
		ip := getInstanceIPFromCloud(rs.cloudProvider, host)
		data = ipCacheEntry{
			ip:         ip,
			lastUpdate: now,
		}
		rs.ipCache[host] = data
	}
	return data.ip
}

func getInstanceIPFromCloud(cloud cloudprovider.Interface, host string) string {
	if cloud == nil {
		return ""
	}
	instances, ok := cloud.Instances()
	if instances == nil || !ok {
		return ""
	}
	addr, err := instances.IPAddress(host)
	if err != nil {
		glog.Errorf("Error getting instance IP for %q: %#v", host, err)
		return ""
	}
	return addr.String()
}

func getPodStatus(pod *api.Pod, minions client.MinionInterface) (api.PodCondition, error) {
	if pod.CurrentState.Host == "" {
		return api.PodPending, nil
	}
	if minions != nil {
		res, err := minions.List()
		if err != nil {
			glog.Errorf("Error listing minions: %v", err)
			return "", err
		}
		found := false
		for _, minion := range res.Items {
			if minion.Name == pod.CurrentState.Host {
				found = true
				break
			}
		}
		if !found {
			return api.PodFailed, nil
		}
	} else {
		glog.Errorf("Unexpected missing minion interface, status may be in-accurate")
	}
	if pod.CurrentState.Info == nil {
		return api.PodPending, nil
	}
	running := 0
	stopped := 0
	unknown := 0
	for _, container := range pod.DesiredState.Manifest.Containers {
		if containerStatus, ok := pod.CurrentState.Info[container.Name]; ok {
			if containerStatus.State.Running != nil {
				running++
			} else if containerStatus.State.Termination != nil {
				stopped++
			} else {
				unknown++
			}
		} else {
			unknown++
		}
	}
	switch {
	case running > 0 && unknown == 0:
		return api.PodRunning, nil
	case running == 0 && stopped > 0 && unknown == 0:
		return api.PodFailed, nil
	case running == 0 && stopped == 0 && unknown > 0:
		return api.PodPending, nil
	default:
		return api.PodPending, nil
	}
}
