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

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/cache"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"

	"github.com/golang/glog"
)

func GetAccessModesAsString(modes []api.PersistentVolumeAccessMode) string {
	modesAsString := ""

	if contains(modes, api.ReadWriteOnce) {
		appendAccessMode(&modesAsString, "RWO")
	}
	if contains(modes, api.ReadOnlyMany) {
		appendAccessMode(&modesAsString, "ROX")
	}
	if contains(modes, api.ReadWriteMany) {
		appendAccessMode(&modesAsString, "RWX")
	}

	return modesAsString
}

func appendAccessMode(modes *string, mode string) {
	if *modes != "" {
		*modes += ","
	}
	*modes += mode
}

func contains(modes []api.PersistentVolumeAccessMode, mode api.PersistentVolumeAccessMode) bool {
	for _, m := range modes {
		if m == mode {
			return true
		}
	}
	return false
}

// ScrubPodVolumeAndWatchUntilCompletion is intended for use with volume Recyclers.  This function will
// save the given Pod to the API and watch it until it completes, fails, or the pod's ActiveDeadlineSeconds is exceeded, whichever comes first.
// An attempt to delete a scrubber pod is always attempted before returning.
// 	pod - the pod designed by a volume plugin to scrub the volume's contents
//	client - kube client for API operations.
func ScrubPodVolumeAndWatchUntilCompletion(pod *api.Pod, kubeClient client.Interface) error {
	return internalScrubPodVolumeAndWatchUntilCompletion(pod, newScrubberClient(kubeClient))
}

// same as above func comments, except 'scrubberClient' is a narrower pod API interface to ease testing
func internalScrubPodVolumeAndWatchUntilCompletion(pod *api.Pod, scrubberClient scrubberClient) error {
	glog.V(5).Infof("Creating scrubber pod for volume %s\n", pod.Name)
	pod, err := scrubberClient.CreatePod(pod)
	if err != nil {
		return fmt.Errorf("Unexpected error creating a pod to scrub volume %s:  %+v\n", pod.Name, err)
	}

	defer scrubberClient.DeletePod(pod.Name, pod.Namespace)

	stopChannel := make(chan struct{})
	defer close(stopChannel)
	nextPod := scrubberClient.WatchPod(pod.Name, pod.Namespace, pod.ResourceVersion, stopChannel)

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

// scrubberClient abstracts access to a Pod by providing a narrower interface.
// this makes it easier to mock a client for testing
type scrubberClient interface {
	CreatePod(pod *api.Pod) (*api.Pod, error)
	GetPod(name, namespace string) (*api.Pod, error)
	DeletePod(name, namespace string) error
	WatchPod(name, namespace, resourceVersion string, stopChannel chan struct{}) func() *api.Pod
}

func newScrubberClient(client client.Interface) scrubberClient {
	return &realScrubberClient{client}
}

type realScrubberClient struct {
	client client.Interface
}

func (c *realScrubberClient) CreatePod(pod *api.Pod) (*api.Pod, error) {
	return c.client.Pods(pod.Namespace).Create(pod)
}

func (c *realScrubberClient) GetPod(name, namespace string) (*api.Pod, error) {
	return c.client.Pods(namespace).Get(name)
}

func (c *realScrubberClient) DeletePod(name, namespace string) error {
	return c.client.Pods(namespace).Delete(name, nil)
}

// WatchPod returns a ListWatch for watching a pod.  The stopChannel is used
// to close the reflector backing the watch.  The caller is responsible for derring a close on the channel to
// stop the reflector.
func (c *realScrubberClient) WatchPod(name, namespace, resourceVersion string, stopChannel chan struct{}) func() *api.Pod {
	fieldSelector, _ := fields.ParseSelector("metadata.name=" + name)

	podLW := &cache.ListWatch{
		ListFunc: func() (runtime.Object, error) {
			return c.client.Pods(namespace).List(labels.Everything(), fieldSelector)
		},
		WatchFunc: func(resourceVersion string) (watch.Interface, error) {
			return c.client.Pods(namespace).Watch(labels.Everything(), fieldSelector, resourceVersion)
		},
	}
	queue := cache.NewFIFO(cache.MetaNamespaceKeyFunc)
	cache.NewReflector(podLW, &api.Pod{}, queue, 1*time.Minute).RunUntil(stopChannel)

	return func() *api.Pod {
		obj := queue.Pop()
		return obj.(*api.Pod)
	}
}
