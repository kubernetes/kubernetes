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

package v1

import (
	"fmt"

	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/watch"
)

// LimitRangesNamespacer has methods to work with LimitRange resources in a namespace
type LimitRangesNamespacer interface {
	LimitRanges(namespace string) LimitRangeInterface
}

// LimitRangeInterface has methods to work with LimitRange resources.
type LimitRangeInterface interface {
	List(selector labels.Selector) (*v1.LimitRangeList, error)
	Get(name string) (*v1.LimitRange, error)
	Delete(name string) error
	Create(limitRange *v1.LimitRange) (*v1.LimitRange, error)
	Update(limitRange *v1.LimitRange) (*v1.LimitRange, error)
	Watch(label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error)
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
func (c *limitRanges) List(selector labels.Selector) (result *v1.LimitRangeList, err error) {
	result = &v1.LimitRangeList{}
	err = c.r.Get().Namespace(c.ns).Resource("limitRanges").LabelsSelectorParam(selector).Do().Into(result)
	return
}

// Get takes the name of the limitRange, and returns the corresponding Pod object, and an error if it occurs
func (c *limitRanges) Get(name string) (result *v1.LimitRange, err error) {
	result = &v1.LimitRange{}
	err = c.r.Get().Namespace(c.ns).Resource("limitRanges").Name(name).Do().Into(result)
	return
}

// Delete takes the name of the limitRange, and returns an error if one occurs
func (c *limitRanges) Delete(name string) error {
	return c.r.Delete().Namespace(c.ns).Resource("limitRanges").Name(name).Do().Error()
}

// Create takes the representation of a limitRange.  Returns the server's representation of the limitRange, and an error, if it occurs.
func (c *limitRanges) Create(limitRange *v1.LimitRange) (result *v1.LimitRange, err error) {
	result = &v1.LimitRange{}
	err = c.r.Post().Namespace(c.ns).Resource("limitRanges").Body(limitRange).Do().Into(result)
	return
}

// Update takes the representation of a limitRange to update.  Returns the server's representation of the limitRange, and an error, if it occurs.
func (c *limitRanges) Update(limitRange *v1.LimitRange) (result *v1.LimitRange, err error) {
	result = &v1.LimitRange{}
	if len(limitRange.ResourceVersion) == 0 {
		err = fmt.Errorf("invalid update object, missing resource version: %v", limitRange)
		return
	}
	err = c.r.Put().Namespace(c.ns).Resource("limitRanges").Name(limitRange.Name).Body(limitRange).Do().Into(result)
	return
}

// Watch returns a watch.Interface that watches the requested resource
func (c *limitRanges) Watch(label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error) {
	return c.r.Get().
		Prefix("watch").
		Namespace(c.ns).
		Resource("limitRanges").
		Param("resourceVersion", resourceVersion).
		LabelsSelectorParam(label).
		FieldsSelectorParam(field).
		Watch()
}
