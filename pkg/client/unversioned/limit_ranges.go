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

package unversioned

import (
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/watch"
)

// LimitRangesNamespacer has methods to work with LimitRange resources in a namespace
type LimitRangesNamespacer interface {
	LimitRanges(namespace string) LimitRangeInterface
}

// LimitRangeInterface has methods to work with LimitRange resources.
type LimitRangeInterface interface {
	List(opts api.ListOptions) (*api.LimitRangeList, error)
	Get(name string) (*api.LimitRange, error)
	Delete(name string) error
	Create(limitRange *api.LimitRange) (*api.LimitRange, error)
	Update(limitRange *api.LimitRange) (*api.LimitRange, error)
	Watch(opts api.ListOptions) (watch.Interface, error)
}

// limitRanges implements LimitRangesNamespacer interface
type limitRanges struct {
	r  *Client
	ns string
}

// newLimitRanges returns a limitRanges
func newLimitRanges(c *Client, namespace string) *limitRanges {
	return &limitRanges{
		r:  c,
		ns: namespace,
	}
}

// List takes a selector, and returns the list of limitRanges that match that selector.
func (c *limitRanges) List(opts api.ListOptions) (result *api.LimitRangeList, err error) {
	result = &api.LimitRangeList{}
	err = c.r.Get().Namespace(c.ns).Resource("limitRanges").VersionedParams(&opts, api.ParameterCodec).Do().Into(result)
	return
}

// Get takes the name of the limitRange, and returns the corresponding Pod object, and an error if it occurs
func (c *limitRanges) Get(name string) (result *api.LimitRange, err error) {
	result = &api.LimitRange{}
	err = c.r.Get().Namespace(c.ns).Resource("limitRanges").Name(name).Do().Into(result)
	return
}

// Delete takes the name of the limitRange, and returns an error if one occurs
func (c *limitRanges) Delete(name string) error {
	return c.r.Delete().Namespace(c.ns).Resource("limitRanges").Name(name).Do().Error()
}

// Create takes the representation of a limitRange.  Returns the server's representation of the limitRange, and an error, if it occurs.
func (c *limitRanges) Create(limitRange *api.LimitRange) (result *api.LimitRange, err error) {
	result = &api.LimitRange{}
	err = c.r.Post().Namespace(c.ns).Resource("limitRanges").Body(limitRange).Do().Into(result)
	return
}

// Update takes the representation of a limitRange to update.  Returns the server's representation of the limitRange, and an error, if it occurs.
func (c *limitRanges) Update(limitRange *api.LimitRange) (result *api.LimitRange, err error) {
	result = &api.LimitRange{}
	err = c.r.Put().Namespace(c.ns).Resource("limitRanges").Name(limitRange.Name).Body(limitRange).Do().Into(result)
	return
}

// Watch returns a watch.Interface that watches the requested resource
func (c *limitRanges) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.r.Get().
		Prefix("watch").
		Namespace(c.ns).
		Resource("limitRanges").
		VersionedParams(&opts, api.ParameterCodec).
		Watch()
}
