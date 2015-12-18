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

// LimitRangeNamespacer has methods to work with LimitRange resources in a namespace
type LimitRangeNamespacer interface {
	LimitRanges(namespace string) LimitRangeInterface
}

// LimitRangeInterface has methods to work with LimitRange resources.
type LimitRangeInterface interface {
	Create(*api.LimitRange) (*api.LimitRange, error)
	Update(*api.LimitRange) (*api.LimitRange, error)
	Delete(name string, options *api.DeleteOptions) error
	Get(name string) (*api.LimitRange, error)
	List(opts unversioned.ListOptions) (*api.LimitRangeList, error)
	Watch(opts unversioned.ListOptions) (watch.Interface, error)
}

// limitRanges implements LimitRangeInterface
type limitRanges struct {
	client *LegacyClient
	ns     string
}

// newLimitRanges returns a LimitRanges
func newLimitRanges(c *LegacyClient, namespace string) *limitRanges {
	return &limitRanges{
		client: c,
		ns:     namespace,
	}
}

// Create takes the representation of a limitRange and creates it.  Returns the server's representation of the limitRange, and an error, if there is any.
func (c *limitRanges) Create(limitRange *api.LimitRange) (result *api.LimitRange, err error) {
	result = &api.LimitRange{}
	err = c.client.Post().
		Namespace(c.ns).
		Resource("limitRanges").
		Body(limitRange).
		Do().
		Into(result)
	return
}

// Update takes the representation of a limitRange and updates it. Returns the server's representation of the limitRange, and an error, if there is any.
func (c *limitRanges) Update(limitRange *api.LimitRange) (result *api.LimitRange, err error) {
	result = &api.LimitRange{}
	err = c.client.Put().
		Namespace(c.ns).
		Resource("limitRanges").
		Name(limitRange.Name).
		Body(limitRange).
		Do().
		Into(result)
	return
}

// Delete takes name of the limitRange and deletes it. Returns an error if one occurs.
func (c *limitRanges) Delete(name string, options *api.DeleteOptions) error {
	if options == nil {
		return c.client.Delete().Namespace(c.ns).Resource("limitRanges").Name(name).Do().Error()
	}
	body, err := api.Scheme.EncodeToVersion(options, c.client.APIVersion().String())
	if err != nil {
		return err
	}
	return c.client.Delete().
		Namespace(c.ns).
		Resource("limitRanges").
		Name(name).
		Body(body).
		Do().
		Error()
}

// Get takes name of the limitRange, and returns the corresponding limitRange object, and an error if there is any.
func (c *limitRanges) Get(name string) (result *api.LimitRange, err error) {
	result = &api.LimitRange{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("limitRanges").
		Name(name).
		Do().
		Into(result)
	return
}

// List takes label and field selectors, and returns the list of LimitRanges that match those selectors.
func (c *limitRanges) List(opts unversioned.ListOptions) (result *api.LimitRangeList, err error) {
	result = &api.LimitRangeList{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("limitRanges").
		VersionedParams(&opts, api.Scheme).
		Do().
		Into(result)
	return
}

// Watch returns a watch.Interface that watches the requested limitRanges.
func (c *limitRanges) Watch(opts unversioned.ListOptions) (watch.Interface, error) {
	return c.client.Get().
		Prefix("watch").
		Namespace(c.ns).
		Resource("limitRanges").
		VersionedParams(&opts, api.Scheme).
		Watch()
}
