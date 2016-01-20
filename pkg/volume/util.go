/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package volume

import (
	"fmt"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/cache"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/watch"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api/resource"
)

// RecycleVolumeByWatchingPodUntilCompletion is intended for use with volume Recyclers.  This function will
// save the given Pod to the API and watch it until it completes, fails, or the pod's ActiveDeadlineSeconds is exceeded, whichever comes first.
// An attempt to delete a recycler pod is always attempted before returning.
// 	pod - the pod designed by a volume plugin to recycle the volume
//	client - kube client for API operations.
func RecycleVolumeByWatchingPodUntilCompletion(pod *api.Pod, kubeClient client.Interface) error {
	return internalRecycleVolumeByWatchingPodUntilCompletion(pod, newRecyclerClient(kubeClient))
}

// same as above func comments, except 'recyclerClient' is a narrower pod API interface to ease testing
func internalRecycleVolumeByWatchingPodUntilCompletion(pod *api.Pod, recyclerClient recyclerClient) error {
	glog.V(5).Infof("Creating recycler pod for volume %s\n", pod.Name)
	pod, err := recyclerClient.CreatePod(pod)
	if err != nil {
		return fmt.Errorf("Unexpected error creating recycler pod:  %+v\n", err)
	}

	defer recyclerClient.DeletePod(pod.Name, pod.Namespace)

	stopChannel := make(chan struct{})
	defer close(stopChannel)
	nextPod := recyclerClient.WatchPod(pod.Name, pod.Namespace, pod.ResourceVersion, stopChannel)

	for {
		watchedPod := nextPod()
		if watchedPod.Status.Phase == api.PodSucceeded {
			// volume.Recycle() returns nil on success, else error
			return nil
		}
		if watchedPod.Status.Phase == api.PodFailed {
			// volume.Recycle() returns nil on success, else error
			if watchedPod.Status.Message != "" {
				return fmt.Errorf(watchedPod.Status.Message)
			} else {
				return fmt.Errorf("Pod failed, pod.Status.Message unknown.")
			}
		}
	}
}

// recyclerClient abstracts access to a Pod by providing a narrower interface.
// this makes it easier to mock a client for testing
type recyclerClient interface {
	CreatePod(pod *api.Pod) (*api.Pod, error)
	GetPod(name, namespace string) (*api.Pod, error)
	DeletePod(name, namespace string) error
	WatchPod(name, namespace, resourceVersion string, stopChannel chan struct{}) func() *api.Pod
}

func newRecyclerClient(client client.Interface) recyclerClient {
	return &realRecyclerClient{client}
}

type realRecyclerClient struct {
	client client.Interface
}

func (c *realRecyclerClient) CreatePod(pod *api.Pod) (*api.Pod, error) {
	return c.client.Pods(pod.Namespace).Create(pod)
}

func (c *realRecyclerClient) GetPod(name, namespace string) (*api.Pod, error) {
	return c.client.Pods(namespace).Get(name)
}

func (c *realRecyclerClient) DeletePod(name, namespace string) error {
	return c.client.Pods(namespace).Delete(name, nil)
}

// WatchPod returns a ListWatch for watching a pod.  The stopChannel is used
// to close the reflector backing the watch.  The caller is responsible for derring a close on the channel to
// stop the reflector.
func (c *realRecyclerClient) WatchPod(name, namespace, resourceVersion string, stopChannel chan struct{}) func() *api.Pod {
	fieldSelector, _ := fields.ParseSelector("metadata.name=" + name)

	podLW := &cache.ListWatch{
		ListFunc: func(options api.ListOptions) (runtime.Object, error) {
			options.FieldSelector = fieldSelector
			return c.client.Pods(namespace).List(options)
		},
		WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
			options.FieldSelector = fieldSelector
			return c.client.Pods(namespace).Watch(options)
		},
	}
	queue := cache.NewFIFO(cache.MetaNamespaceKeyFunc)
	cache.NewReflector(podLW, &api.Pod{}, queue, 1*time.Minute).RunUntil(stopChannel)

	return func() *api.Pod {
		obj := queue.Pop()
		return obj.(*api.Pod)
	}
}

// CalculateTimeoutForVolume calculates time for a Recycler pod to complete a recycle operation.
// The calculation and return value is either the minimumTimeout or the timeoutIncrement per Gi of storage size, whichever is greater.
func CalculateTimeoutForVolume(minimumTimeout, timeoutIncrement int, pv *api.PersistentVolume) int64 {
	giQty := resource.MustParse("1Gi")
	pvQty := pv.Spec.Capacity[api.ResourceStorage]
	giSize := giQty.Value()
	pvSize := pvQty.Value()
	timeout := (pvSize / giSize) * int64(timeoutIncrement)
	if timeout < int64(minimumTimeout) {
		return int64(minimumTimeout)
	} else {
		return timeout
	}
}

// RoundUpSize calculates how many allocation units are needed to accomodate
// a volume of given size. E.g. when user wants 1500MiB volume, while AWS EBS
// allocates volumes in gibibyte-sized chunks,
// RoundUpSize(1500 * 1024*1024, 1024*1024*1024) returns '2'
// (2 GiB is the smallest allocatable volume that can hold 1500MiB)
func RoundUpSize(volumeSizeBytes int64, allocationUnitBytes int64) int64 {
	return (volumeSizeBytes + allocationUnitBytes - 1) / allocationUnitBytes
}
