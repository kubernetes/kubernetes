/*
Copyright 2016 The Kubernetes Authors.

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

package v1beta1

import (
	api "k8s.io/client-go/1.4/pkg/api"
	v1beta1 "k8s.io/client-go/1.4/pkg/apis/extensions/v1beta1"
	watch "k8s.io/client-go/1.4/pkg/watch"
)

// HorizontalPodAutoscalersGetter has a method to return a HorizontalPodAutoscalerInterface.
// A group's client should implement this interface.
type HorizontalPodAutoscalersGetter interface {
	HorizontalPodAutoscalers(namespace string) HorizontalPodAutoscalerInterface
}

// HorizontalPodAutoscalerInterface has methods to work with HorizontalPodAutoscaler resources.
type HorizontalPodAutoscalerInterface interface {
	Create(*v1beta1.HorizontalPodAutoscaler) (*v1beta1.HorizontalPodAutoscaler, error)
	Update(*v1beta1.HorizontalPodAutoscaler) (*v1beta1.HorizontalPodAutoscaler, error)
	UpdateStatus(*v1beta1.HorizontalPodAutoscaler) (*v1beta1.HorizontalPodAutoscaler, error)
	Delete(name string, options *api.DeleteOptions) error
	DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error
	Get(name string) (*v1beta1.HorizontalPodAutoscaler, error)
	List(opts api.ListOptions) (*v1beta1.HorizontalPodAutoscalerList, error)
	Watch(opts api.ListOptions) (watch.Interface, error)
	Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *v1beta1.HorizontalPodAutoscaler, err error)
	HorizontalPodAutoscalerExpansion
}

// horizontalPodAutoscalers implements HorizontalPodAutoscalerInterface
type horizontalPodAutoscalers struct {
	client *ExtensionsClient
	ns     string
}

// newHorizontalPodAutoscalers returns a HorizontalPodAutoscalers
func newHorizontalPodAutoscalers(c *ExtensionsClient, namespace string) *horizontalPodAutoscalers {
	return &horizontalPodAutoscalers{
		client: c,
		ns:     namespace,
	}
}

// Create takes the representation of a horizontalPodAutoscaler and creates it.  Returns the server's representation of the horizontalPodAutoscaler, and an error, if there is any.
func (c *horizontalPodAutoscalers) Create(horizontalPodAutoscaler *v1beta1.HorizontalPodAutoscaler) (result *v1beta1.HorizontalPodAutoscaler, err error) {
	result = &v1beta1.HorizontalPodAutoscaler{}
	err = c.client.Post().
		Namespace(c.ns).
		Resource("horizontalpodautoscalers").
		Body(horizontalPodAutoscaler).
		Do().
		Into(result)
	return
}

// Update takes the representation of a horizontalPodAutoscaler and updates it. Returns the server's representation of the horizontalPodAutoscaler, and an error, if there is any.
func (c *horizontalPodAutoscalers) Update(horizontalPodAutoscaler *v1beta1.HorizontalPodAutoscaler) (result *v1beta1.HorizontalPodAutoscaler, err error) {
	result = &v1beta1.HorizontalPodAutoscaler{}
	err = c.client.Put().
		Namespace(c.ns).
		Resource("horizontalpodautoscalers").
		Name(horizontalPodAutoscaler.Name).
		Body(horizontalPodAutoscaler).
		Do().
		Into(result)
	return
}

func (c *horizontalPodAutoscalers) UpdateStatus(horizontalPodAutoscaler *v1beta1.HorizontalPodAutoscaler) (result *v1beta1.HorizontalPodAutoscaler, err error) {
	result = &v1beta1.HorizontalPodAutoscaler{}
	err = c.client.Put().
		Namespace(c.ns).
		Resource("horizontalpodautoscalers").
		Name(horizontalPodAutoscaler.Name).
		SubResource("status").
		Body(horizontalPodAutoscaler).
		Do().
		Into(result)
	return
}

// Delete takes name of the horizontalPodAutoscaler and deletes it. Returns an error if one occurs.
func (c *horizontalPodAutoscalers) Delete(name string, options *api.DeleteOptions) error {
	return c.client.Delete().
		Namespace(c.ns).
		Resource("horizontalpodautoscalers").
		Name(name).
		Body(options).
		Do().
		Error()
}

// DeleteCollection deletes a collection of objects.
func (c *horizontalPodAutoscalers) DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error {
	return c.client.Delete().
		Namespace(c.ns).
		Resource("horizontalpodautoscalers").
		VersionedParams(&listOptions, api.ParameterCodec).
		Body(options).
		Do().
		Error()
}

// Get takes name of the horizontalPodAutoscaler, and returns the corresponding horizontalPodAutoscaler object, and an error if there is any.
func (c *horizontalPodAutoscalers) Get(name string) (result *v1beta1.HorizontalPodAutoscaler, err error) {
	result = &v1beta1.HorizontalPodAutoscaler{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("horizontalpodautoscalers").
		Name(name).
		Do().
		Into(result)
	return
}

// List takes label and field selectors, and returns the list of HorizontalPodAutoscalers that match those selectors.
func (c *horizontalPodAutoscalers) List(opts api.ListOptions) (result *v1beta1.HorizontalPodAutoscalerList, err error) {
	result = &v1beta1.HorizontalPodAutoscalerList{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("horizontalpodautoscalers").
		VersionedParams(&opts, api.ParameterCodec).
		Do().
		Into(result)
	return
}

// Watch returns a watch.Interface that watches the requested horizontalPodAutoscalers.
func (c *horizontalPodAutoscalers) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.client.Get().
		Prefix("watch").
		Namespace(c.ns).
		Resource("horizontalpodautoscalers").
		VersionedParams(&opts, api.ParameterCodec).
		Watch()
}

// Patch applies the patch and returns the patched horizontalPodAutoscaler.
func (c *horizontalPodAutoscalers) Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *v1beta1.HorizontalPodAutoscaler, err error) {
	result = &v1beta1.HorizontalPodAutoscaler{}
	err = c.client.Patch(pt).
		Namespace(c.ns).
		Resource("horizontalpodautoscalers").
		SubResource(subresources...).
		Name(name).
		Body(data).
		Do().
		Into(result)
	return
}
