/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package unversioned

import (
	api "k8s.io/kubernetes/pkg/api"
	unversioned "k8s.io/kubernetes/pkg/api/unversioned"
	watch "k8s.io/kubernetes/pkg/watch"
)

// PodNamespacer has methods to work with Pod resources in a namespace
type PodNamespacer interface {
	Pods(namespace string) PodInterface
}

// PodInterface has methods to work with Pod resources.
type PodInterface interface {
	Create(*api.Pod) (*api.Pod, error)
	Update(*api.Pod) (*api.Pod, error)
	Delete(name string, options *api.DeleteOptions) error
	Get(name string) (*api.Pod, error)
	List(opts unversioned.ListOptions) (*api.PodList, error)
	Watch(opts unversioned.ListOptions) (watch.Interface, error)
}

// pods implements PodInterface
type pods struct {
	client *LegacyClient
	ns     string
}

// newPods returns a Pods
func newPods(c *LegacyClient, namespace string) *pods {
	return &pods{
		client: c,
		ns:     namespace,
	}
}

// Create takes the representation of a pod and creates it.  Returns the server's representation of the pod, and an error, if there is any.
func (c *pods) Create(pod *api.Pod) (result *api.Pod, err error) {
	result = &api.Pod{}
	err = c.client.Post().
		Namespace(c.ns).
		Resource("pods").
		Body(pod).
		Do().
		Into(result)
	return
}

// Update takes the representation of a pod and updates it. Returns the server's representation of the pod, and an error, if there is any.
func (c *pods) Update(pod *api.Pod) (result *api.Pod, err error) {
	result = &api.Pod{}
	err = c.client.Put().
		Namespace(c.ns).
		Resource("pods").
		Name(pod.Name).
		Body(pod).
		Do().
		Into(result)
	return
}

// Delete takes name of the pod and deletes it. Returns an error if one occurs.
func (c *pods) Delete(name string, options *api.DeleteOptions) error {
	if options == nil {
		return c.client.Delete().Namespace(c.ns).Resource("pods").Name(name).Do().Error()
	}
	body, err := api.Scheme.EncodeToVersion(options, c.client.APIVersion().String())
	if err != nil {
		return err
	}
	return c.client.Delete().
		Namespace(c.ns).
		Resource("pods").
		Name(name).
		Body(body).
		Do().
		Error()
}

// Get takes name of the pod, and returns the corresponding pod object, and an error if there is any.
func (c *pods) Get(name string) (result *api.Pod, err error) {
	result = &api.Pod{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("pods").
		Name(name).
		Do().
		Into(result)
	return
}

// List takes label and field selectors, and returns the list of Pods that match those selectors.
func (c *pods) List(opts unversioned.ListOptions) (result *api.PodList, err error) {
	result = &api.PodList{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("pods").
		VersionedParams(&opts, api.Scheme).
		Do().
		Into(result)
	return
}

// Watch returns a watch.Interface that watches the requested pods.
func (c *pods) Watch(opts unversioned.ListOptions) (watch.Interface, error) {
	return c.client.Get().
		Prefix("watch").
		Namespace(c.ns).
		Resource("pods").
		VersionedParams(&opts, api.Scheme).
		Watch()
}
