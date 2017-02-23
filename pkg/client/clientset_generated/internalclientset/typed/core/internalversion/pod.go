/*
Copyright 2017 The Kubernetes Authors.

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

package internalversion

import (
	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	types "k8s.io/apimachinery/pkg/types"
	watch "k8s.io/apimachinery/pkg/watch"
	rest "k8s.io/client-go/rest"
	api "k8s.io/kubernetes/pkg/api"
	scheme "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/scheme"
)

// PodsGetter has a method to return a PodInterface.
// A group's client should implement this interface.
type PodsGetter interface {
	Pods(namespace string) PodInterface
}

// PodInterface has methods to work with Pod resources.
type PodInterface interface {
	Create(*api.Pod) (*api.Pod, error)
	Update(*api.Pod) (*api.Pod, error)
	UpdateStatus(*api.Pod) (*api.Pod, error)
	Delete(name string, options *v1.DeleteOptions) error
	DeleteCollection(options *v1.DeleteOptions, listOptions v1.ListOptions) error
	Get(name string, options v1.GetOptions) (*api.Pod, error)
	List(opts v1.ListOptions) (*api.PodList, error)
	Watch(opts v1.ListOptions) (watch.Interface, error)
	Patch(name string, pt types.PatchType, data []byte, subresources ...string) (result *api.Pod, err error)
	PodExpansion
}

// pods implements PodInterface
type pods struct {
	client rest.Interface
	ns     string
}

// newPods returns a Pods
func newPods(c *CoreClient, namespace string) *pods {
	return &pods{
		client: c.RESTClient(),
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

// UpdateStatus was generated because the type contains a Status member.
// Add a +genclientstatus=false comment above the type to avoid generating UpdateStatus().

func (c *pods) UpdateStatus(pod *api.Pod) (result *api.Pod, err error) {
	result = &api.Pod{}
	err = c.client.Put().
		Namespace(c.ns).
		Resource("pods").
		Name(pod.Name).
		SubResource("status").
		Body(pod).
		Do().
		Into(result)
	return
}

// Delete takes name of the pod and deletes it. Returns an error if one occurs.
func (c *pods) Delete(name string, options *v1.DeleteOptions) error {
	return c.client.Delete().
		Namespace(c.ns).
		Resource("pods").
		Name(name).
		Body(options).
		Do().
		Error()
}

// DeleteCollection deletes a collection of objects.
func (c *pods) DeleteCollection(options *v1.DeleteOptions, listOptions v1.ListOptions) error {
	return c.client.Delete().
		Namespace(c.ns).
		Resource("pods").
		VersionedParams(&listOptions, scheme.ParameterCodec).
		Body(options).
		Do().
		Error()
}

// Get takes name of the pod, and returns the corresponding pod object, and an error if there is any.
func (c *pods) Get(name string, options v1.GetOptions) (result *api.Pod, err error) {
	result = &api.Pod{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("pods").
		Name(name).
		VersionedParams(&options, scheme.ParameterCodec).
		Do().
		Into(result)
	return
}

// List takes label and field selectors, and returns the list of Pods that match those selectors.
func (c *pods) List(opts v1.ListOptions) (result *api.PodList, err error) {
	result = &api.PodList{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("pods").
		VersionedParams(&opts, scheme.ParameterCodec).
		Do().
		Into(result)
	return
}

// Watch returns a watch.Interface that watches the requested pods.
func (c *pods) Watch(opts v1.ListOptions) (watch.Interface, error) {
	opts.Watch = true
	return c.client.Get().
		Namespace(c.ns).
		Resource("pods").
		VersionedParams(&opts, scheme.ParameterCodec).
		Watch()
}

// Patch applies the patch and returns the patched pod.
func (c *pods) Patch(name string, pt types.PatchType, data []byte, subresources ...string) (result *api.Pod, err error) {
	result = &api.Pod{}
	err = c.client.Patch(pt).
		Namespace(c.ns).
		Resource("pods").
		SubResource(subresources...).
		Name(name).
		Body(data).
		Do().
		Into(result)
	return
}
