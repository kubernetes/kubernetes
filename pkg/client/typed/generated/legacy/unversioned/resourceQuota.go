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

// ResourceQuotaNamespacer has methods to work with ResourceQuota resources in a namespace
type ResourceQuotaNamespacer interface {
	ResourceQuotas(namespace string) ResourceQuotaInterface
}

// ResourceQuotaInterface has methods to work with ResourceQuota resources.
type ResourceQuotaInterface interface {
	Create(*api.ResourceQuota) (*api.ResourceQuota, error)
	Update(*api.ResourceQuota) (*api.ResourceQuota, error)
	Delete(name string, options *api.DeleteOptions) error
	Get(name string) (*api.ResourceQuota, error)
	List(opts unversioned.ListOptions) (*api.ResourceQuotaList, error)
	Watch(opts unversioned.ListOptions) (watch.Interface, error)
}

// resourceQuotas implements ResourceQuotaInterface
type resourceQuotas struct {
	client *LegacyClient
	ns     string
}

// newResourceQuotas returns a ResourceQuotas
func newResourceQuotas(c *LegacyClient, namespace string) *resourceQuotas {
	return &resourceQuotas{
		client: c,
		ns:     namespace,
	}
}

// Create takes the representation of a resourceQuota and creates it.  Returns the server's representation of the resourceQuota, and an error, if there is any.
func (c *resourceQuotas) Create(resourceQuota *api.ResourceQuota) (result *api.ResourceQuota, err error) {
	result = &api.ResourceQuota{}
	err = c.client.Post().
		Namespace(c.ns).
		Resource("resourceQuotas").
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
		Resource("resourceQuotas").
		Name(resourceQuota.Name).
		Body(resourceQuota).
		Do().
		Into(result)
	return
}

// Delete takes name of the resourceQuota and deletes it. Returns an error if one occurs.
func (c *resourceQuotas) Delete(name string, options *api.DeleteOptions) error {
	if options == nil {
		return c.client.Delete().Namespace(c.ns).Resource("resourceQuotas").Name(name).Do().Error()
	}
	body, err := api.Scheme.EncodeToVersion(options, c.client.APIVersion().String())
	if err != nil {
		return err
	}
	return c.client.Delete().
		Namespace(c.ns).
		Resource("resourceQuotas").
		Name(name).
		Body(body).
		Do().
		Error()
}

// Get takes name of the resourceQuota, and returns the corresponding resourceQuota object, and an error if there is any.
func (c *resourceQuotas) Get(name string) (result *api.ResourceQuota, err error) {
	result = &api.ResourceQuota{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("resourceQuotas").
		Name(name).
		Do().
		Into(result)
	return
}

// List takes label and field selectors, and returns the list of ResourceQuotas that match those selectors.
func (c *resourceQuotas) List(opts unversioned.ListOptions) (result *api.ResourceQuotaList, err error) {
	result = &api.ResourceQuotaList{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("resourceQuotas").
		VersionedParams(&opts, api.Scheme).
		Do().
		Into(result)
	return
}

// Watch returns a watch.Interface that watches the requested resourceQuotas.
func (c *resourceQuotas) Watch(opts unversioned.ListOptions) (watch.Interface, error) {
	return c.client.Get().
		Prefix("watch").
		Namespace(c.ns).
		Resource("resourceQuotas").
		VersionedParams(&opts, api.Scheme).
		Watch()
}
