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

package client

import (
	"fmt"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
)

// PodsNamespacer has methods to work with Pod resources in a namespace
type PodsNamespacer interface {
	Pods(namespace string) PodInterface
}

// PodInterface has methods to work with Pod resources.
type PodInterface interface {
	List(selector labels.Selector) (*api.PodList, error)
	Get(name string) (*api.Pod, error)
	Delete(name string) error
	Create(pod *api.Pod) (*api.Pod, error)
	Update(pod *api.Pod) (*api.Pod, error)
}

// pods implements PodsNamespacer interface
type pods struct {
	r  *Client
	ns string
}

// newPods returns a pods
func newPods(c *Client, namespace string) *pods {
	return &pods{
		r:  c,
		ns: namespace,
	}
}

// ListPods takes a selector, and returns the list of pods that match that selector.
func (c *pods) List(selector labels.Selector) (result *api.PodList, err error) {
	result = &api.PodList{}
	err = c.r.Get().Namespace(c.ns).Path("pods").SelectorParam("labels", selector).Do().Into(result)
	return
}

// GetPod takes the name of the pod, and returns the corresponding Pod object, and an error if it occurs
func (c *pods) Get(name string) (result *api.Pod, err error) {
	result = &api.Pod{}
	err = c.r.Get().Namespace(c.ns).Path("pods").Path(name).Do().Into(result)
	return
}

// DeletePod takes the name of the pod, and returns an error if one occurs
func (c *pods) Delete(name string) error {
	return c.r.Delete().Namespace(c.ns).Path("pods").Path(name).Do().Error()
}

// CreatePod takes the representation of a pod.  Returns the server's representation of the pod, and an error, if it occurs.
func (c *pods) Create(pod *api.Pod) (result *api.Pod, err error) {
	result = &api.Pod{}
	err = c.r.Post().Namespace(c.ns).Path("pods").Body(pod).Do().Into(result)
	return
}

// UpdatePod takes the representation of a pod to update.  Returns the server's representation of the pod, and an error, if it occurs.
func (c *pods) Update(pod *api.Pod) (result *api.Pod, err error) {
	result = &api.Pod{}
	if len(pod.ResourceVersion) == 0 {
		err = fmt.Errorf("invalid update object, missing resource version: %v", pod)
		return
	}
	err = c.r.Put().Namespace(c.ns).Path("pods").Path(pod.Name).Body(pod).Do().Into(result)
	return
}
