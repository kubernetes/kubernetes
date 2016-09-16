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

package v1

import (
	api "k8s.io/client-go/1.4/pkg/api"
	v1 "k8s.io/client-go/1.4/pkg/api/v1"
	watch "k8s.io/client-go/1.4/pkg/watch"
)

// ResourceQuotasGetter has a method to return a ResourceQuotaInterface.
// A group's client should implement this interface.
type ResourceQuotasGetter interface {
	ResourceQuotas(namespace string) ResourceQuotaInterface
}

// ResourceQuotaInterface has methods to work with ResourceQuota resources.
type ResourceQuotaInterface interface {
	Create(*v1.ResourceQuota) (*v1.ResourceQuota, error)
	Update(*v1.ResourceQuota) (*v1.ResourceQuota, error)
	UpdateStatus(*v1.ResourceQuota) (*v1.ResourceQuota, error)
	Delete(name string, options *api.DeleteOptions) error
	DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error
	Get(name string) (*v1.ResourceQuota, error)
	List(opts api.ListOptions) (*v1.ResourceQuotaList, error)
	Watch(opts api.ListOptions) (watch.Interface, error)
	Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *v1.ResourceQuota, err error)
	ResourceQuotaExpansion
}

// resourceQuotas implements ResourceQuotaInterface
type resourceQuotas struct {
	client *CoreClient
	ns     string
}

// newResourceQuotas returns a ResourceQuotas
func newResourceQuotas(c *CoreClient, namespace string) *resourceQuotas {
	return &resourceQuotas{
		client: c,
		ns:     namespace,
	}
}

// Create takes the representation of a resourceQuota and creates it.  Returns the server's representation of the resourceQuota, and an error, if there is any.
func (c *resourceQuotas) Create(resourceQuota *v1.ResourceQuota) (result *v1.ResourceQuota, err error) {
	result = &v1.ResourceQuota{}
	err = c.client.Post().
		Namespace(c.ns).
		Resource("resourcequotas").
		Body(resourceQuota).
		Do().
		Into(result)
	return
}

// Update takes the representation of a resourceQuota and updates it. Returns the server's representation of the resourceQuota, and an error, if there is any.
func (c *resourceQuotas) Update(resourceQuota *v1.ResourceQuota) (result *v1.ResourceQuota, err error) {
	result = &v1.ResourceQuota{}
	err = c.client.Put().
		Namespace(c.ns).
		Resource("resourcequotas").
		Name(resourceQuota.Name).
		Body(resourceQuota).
		Do().
		Into(result)
	return
}

func (c *resourceQuotas) UpdateStatus(resourceQuota *v1.ResourceQuota) (result *v1.ResourceQuota, err error) {
	result = &v1.ResourceQuota{}
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
func (c *resourceQuotas) Delete(name string, options *api.DeleteOptions) error {
	return c.client.Delete().
		Namespace(c.ns).
		Resource("resourcequotas").
		Name(name).
		Body(options).
		Do().
		Error()
}

// DeleteCollection deletes a collection of objects.
func (c *resourceQuotas) DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error {
	return c.client.Delete().
		Namespace(c.ns).
		Resource("resourcequotas").
		VersionedParams(&listOptions, api.ParameterCodec).
		Body(options).
		Do().
		Error()
}

// Get takes name of the resourceQuota, and returns the corresponding resourceQuota object, and an error if there is any.
func (c *resourceQuotas) Get(name string) (result *v1.ResourceQuota, err error) {
	result = &v1.ResourceQuota{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("resourcequotas").
		Name(name).
		Do().
		Into(result)
	return
}

// List takes label and field selectors, and returns the list of ResourceQuotas that match those selectors.
func (c *resourceQuotas) List(opts api.ListOptions) (result *v1.ResourceQuotaList, err error) {
	result = &v1.ResourceQuotaList{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("resourcequotas").
		VersionedParams(&opts, api.ParameterCodec).
		Do().
		Into(result)
	return
}

// Watch returns a watch.Interface that watches the requested resourceQuotas.
func (c *resourceQuotas) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.client.Get().
		Prefix("watch").
		Namespace(c.ns).
		Resource("resourcequotas").
		VersionedParams(&opts, api.ParameterCodec).
		Watch()
}

// Patch applies the patch and returns the patched resourceQuota.
func (c *resourceQuotas) Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *v1.ResourceQuota, err error) {
	result = &v1.ResourceQuota{}
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
