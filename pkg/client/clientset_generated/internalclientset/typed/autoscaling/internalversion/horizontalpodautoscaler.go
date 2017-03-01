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
	autoscaling "k8s.io/kubernetes/pkg/apis/autoscaling"
	scheme "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/scheme"
)

// HorizontalPodAutoscalersGetter has a method to return a HorizontalPodAutoscalerInterface.
// A group's client should implement this interface.
type HorizontalPodAutoscalersGetter interface {
	HorizontalPodAutoscalers(namespace string) HorizontalPodAutoscalerInterface
}

// HorizontalPodAutoscalerInterface has methods to work with HorizontalPodAutoscaler resources.
type HorizontalPodAutoscalerInterface interface {
	Create(*autoscaling.HorizontalPodAutoscaler) (*autoscaling.HorizontalPodAutoscaler, error)
	Update(*autoscaling.HorizontalPodAutoscaler) (*autoscaling.HorizontalPodAutoscaler, error)
	UpdateStatus(*autoscaling.HorizontalPodAutoscaler) (*autoscaling.HorizontalPodAutoscaler, error)
	Delete(name string, options *v1.DeleteOptions) error
	DeleteCollection(options *v1.DeleteOptions, listOptions v1.ListOptions) error
	Get(name string, options v1.GetOptions) (*autoscaling.HorizontalPodAutoscaler, error)
	List(opts v1.ListOptions) (*autoscaling.HorizontalPodAutoscalerList, error)
	Watch(opts v1.ListOptions) (watch.Interface, error)
	Patch(name string, pt types.PatchType, data []byte, subresources ...string) (result *autoscaling.HorizontalPodAutoscaler, err error)
	HorizontalPodAutoscalerExpansion
}

// horizontalPodAutoscalers implements HorizontalPodAutoscalerInterface
type horizontalPodAutoscalers struct {
	client rest.Interface
	ns     string
}

// newHorizontalPodAutoscalers returns a HorizontalPodAutoscalers
func newHorizontalPodAutoscalers(c *AutoscalingClient, namespace string) *horizontalPodAutoscalers {
	return &horizontalPodAutoscalers{
		client: c.RESTClient(),
		ns:     namespace,
	}
}

// Create takes the representation of a horizontalPodAutoscaler and creates it.  Returns the server's representation of the horizontalPodAutoscaler, and an error, if there is any.
func (c *horizontalPodAutoscalers) Create(horizontalPodAutoscaler *autoscaling.HorizontalPodAutoscaler) (result *autoscaling.HorizontalPodAutoscaler, err error) {
	result = &autoscaling.HorizontalPodAutoscaler{}
	err = c.client.Post().
		Namespace(c.ns).
		Resource("horizontalpodautoscalers").
		Body(horizontalPodAutoscaler).
		Do().
		Into(result)
	return
}

// Update takes the representation of a horizontalPodAutoscaler and updates it. Returns the server's representation of the horizontalPodAutoscaler, and an error, if there is any.
func (c *horizontalPodAutoscalers) Update(horizontalPodAutoscaler *autoscaling.HorizontalPodAutoscaler) (result *autoscaling.HorizontalPodAutoscaler, err error) {
	result = &autoscaling.HorizontalPodAutoscaler{}
	err = c.client.Put().
		Namespace(c.ns).
		Resource("horizontalpodautoscalers").
		Name(horizontalPodAutoscaler.Name).
		Body(horizontalPodAutoscaler).
		Do().
		Into(result)
	return
}

// UpdateStatus was generated because the type contains a Status member.
// Add a +genclientstatus=false comment above the type to avoid generating UpdateStatus().

func (c *horizontalPodAutoscalers) UpdateStatus(horizontalPodAutoscaler *autoscaling.HorizontalPodAutoscaler) (result *autoscaling.HorizontalPodAutoscaler, err error) {
	result = &autoscaling.HorizontalPodAutoscaler{}
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
func (c *horizontalPodAutoscalers) Delete(name string, options *v1.DeleteOptions) error {
	return c.client.Delete().
		Namespace(c.ns).
		Resource("horizontalpodautoscalers").
		Name(name).
		Body(options).
		Do().
		Error()
}

// DeleteCollection deletes a collection of objects.
func (c *horizontalPodAutoscalers) DeleteCollection(options *v1.DeleteOptions, listOptions v1.ListOptions) error {
	return c.client.Delete().
		Namespace(c.ns).
		Resource("horizontalpodautoscalers").
		VersionedParams(&listOptions, scheme.ParameterCodec).
		Body(options).
		Do().
		Error()
}

// Get takes name of the horizontalPodAutoscaler, and returns the corresponding horizontalPodAutoscaler object, and an error if there is any.
func (c *horizontalPodAutoscalers) Get(name string, options v1.GetOptions) (result *autoscaling.HorizontalPodAutoscaler, err error) {
	result = &autoscaling.HorizontalPodAutoscaler{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("horizontalpodautoscalers").
		Name(name).
		VersionedParams(&options, scheme.ParameterCodec).
		Do().
		Into(result)
	return
}

// List takes label and field selectors, and returns the list of HorizontalPodAutoscalers that match those selectors.
func (c *horizontalPodAutoscalers) List(opts v1.ListOptions) (result *autoscaling.HorizontalPodAutoscalerList, err error) {
	result = &autoscaling.HorizontalPodAutoscalerList{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("horizontalpodautoscalers").
		VersionedParams(&opts, scheme.ParameterCodec).
		Do().
		Into(result)
	return
}

// Watch returns a watch.Interface that watches the requested horizontalPodAutoscalers.
func (c *horizontalPodAutoscalers) Watch(opts v1.ListOptions) (watch.Interface, error) {
	opts.Watch = true
	return c.client.Get().
		Namespace(c.ns).
		Resource("horizontalpodautoscalers").
		VersionedParams(&opts, scheme.ParameterCodec).
		Watch()
}

// Patch applies the patch and returns the patched horizontalPodAutoscaler.
func (c *horizontalPodAutoscalers) Patch(name string, pt types.PatchType, data []byte, subresources ...string) (result *autoscaling.HorizontalPodAutoscaler, err error) {
	result = &autoscaling.HorizontalPodAutoscaler{}
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
