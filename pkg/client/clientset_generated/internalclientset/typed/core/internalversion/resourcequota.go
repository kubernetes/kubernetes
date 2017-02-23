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

// ResourceQuotasGetter has a method to return a ResourceQuotaInterface.
// A group's client should implement this interface.
type ResourceQuotasGetter interface {
	ResourceQuotas(namespace string) ResourceQuotaInterface
}

// ResourceQuotaInterface has methods to work with ResourceQuota resources.
type ResourceQuotaInterface interface {
	Create(*api.ResourceQuota) (*api.ResourceQuota, error)
	Update(*api.ResourceQuota) (*api.ResourceQuota, error)
	UpdateStatus(*api.ResourceQuota) (*api.ResourceQuota, error)
	Delete(name string, options *v1.DeleteOptions) error
	DeleteCollection(options *v1.DeleteOptions, listOptions v1.ListOptions) error
	Get(name string, options v1.GetOptions) (*api.ResourceQuota, error)
	List(opts v1.ListOptions) (*api.ResourceQuotaList, error)
	Watch(opts v1.ListOptions) (watch.Interface, error)
	Patch(name string, pt types.PatchType, data []byte, subresources ...string) (result *api.ResourceQuota, err error)
	ResourceQuotaExpansion
}

// resourceQuotas implements ResourceQuotaInterface
type resourceQuotas struct {
	client rest.Interface
	ns     string
}

// newResourceQuotas returns a ResourceQuotas
func newResourceQuotas(c *CoreClient, namespace string) *resourceQuotas {
	return &resourceQuotas{
		client: c.RESTClient(),
		ns:     namespace,
	}
}

// Create takes the representation of a resourceQuota and creates it.  Returns the server's representation of the resourceQuota, and an error, if there is any.
func (c *resourceQuotas) Create(resourceQuota *api.ResourceQuota) (result *api.ResourceQuota, err error) {
	result = &api.ResourceQuota{}
	err = c.client.Post().
		Namespace(c.ns).
		Resource("resourcequotas").
		Body(resourceQuota).
		Do().
		Into(result)
	return
}

// Update takes the representation of a resourceQuota and updates it. Returns the server's representation of the resourceQuota, and an error, if there is any.
func (c *resourceQuotas) Update(resourceQuota *api.ResourceQuota) (result *api.ResourceQuota, err error) {
	result = &api.ResourceQuota{}
	err = c.client.Put().
		Namespace(c.ns).
		Resource("resourcequotas").
		Name(resourceQuota.Name).
		Body(resourceQuota).
		Do().
		Into(result)
	return
}

// UpdateStatus was generated because the type contains a Status member.
// Add a +genclientstatus=false comment above the type to avoid generating UpdateStatus().

func (c *resourceQuotas) UpdateStatus(resourceQuota *api.ResourceQuota) (result *api.ResourceQuota, err error) {
	result = &api.ResourceQuota{}
	err = c.client.Put().
		Namespace(c.ns).
		Resource("resourcequotas").
		Name(resourceQuota.Name).
		SubResource("status").
		Body(resourceQuota).
		Do().
		Into(result)
	return
}

// Delete takes name of the resourceQuota and deletes it. Returns an error if one occurs.
func (c *resourceQuotas) Delete(name string, options *v1.DeleteOptions) error {
	return c.client.Delete().
		Namespace(c.ns).
		Resource("resourcequotas").
		Name(name).
		Body(options).
		Do().
		Error()
}

// DeleteCollection deletes a collection of objects.
func (c *resourceQuotas) DeleteCollection(options *v1.DeleteOptions, listOptions v1.ListOptions) error {
	return c.client.Delete().
		Namespace(c.ns).
		Resource("resourcequotas").
		VersionedParams(&listOptions, scheme.ParameterCodec).
		Body(options).
		Do().
		Error()
}

// Get takes name of the resourceQuota, and returns the corresponding resourceQuota object, and an error if there is any.
func (c *resourceQuotas) Get(name string, options v1.GetOptions) (result *api.ResourceQuota, err error) {
	result = &api.ResourceQuota{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("resourcequotas").
		Name(name).
		VersionedParams(&options, scheme.ParameterCodec).
		Do().
		Into(result)
	return
}

// List takes label and field selectors, and returns the list of ResourceQuotas that match those selectors.
func (c *resourceQuotas) List(opts v1.ListOptions) (result *api.ResourceQuotaList, err error) {
	result = &api.ResourceQuotaList{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("resourcequotas").
		VersionedParams(&opts, scheme.ParameterCodec).
		Do().
		Into(result)
	return
}

// Watch returns a watch.Interface that watches the requested resourceQuotas.
func (c *resourceQuotas) Watch(opts v1.ListOptions) (watch.Interface, error) {
	opts.Watch = true
	return c.client.Get().
		Namespace(c.ns).
		Resource("resourcequotas").
		VersionedParams(&opts, scheme.ParameterCodec).
		Watch()
}

// Patch applies the patch and returns the patched resourceQuota.
func (c *resourceQuotas) Patch(name string, pt types.PatchType, data []byte, subresources ...string) (result *api.ResourceQuota, err error) {
	result = &api.ResourceQuota{}
	err = c.client.Patch(pt).
		Namespace(c.ns).
		Resource("resourcequotas").
		SubResource(subresources...).
		Name(name).
		Body(data).
		Do().
		Into(result)
	return
}
