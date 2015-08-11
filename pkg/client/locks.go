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

package client

import (
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/watch"
)

// LockNamespacer has methods to work with Lock resources in a namespace
type LockNamespacer interface {
	Locks(namespace string) LockInterface
}

// LockInterface has methods to work with Lock resources.
type LockInterface interface {
	List(selector labels.Selector) (*api.LockList, error)
	Get(name string) (*api.Lock, error)
	Delete(name string) error
	Create(limitRange *api.Lock) (*api.Lock, error)
	Update(limitRange *api.Lock) (*api.Lock, error)
	Watch(label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error)
}

// locks implements LockNamespacer interface
type locks struct {
	r  *Client
	ns string
}

// newLocks returns a lock
func newLocks(c *Client, namespace string) *locks {
	return &locks{
		r:  c,
		ns: namespace,
	}
}

// List takes a selector, and returns the list of locks that match that selector.
func (c *locks) List(selector labels.Selector) (result *api.LockList, err error) {
	result = &api.LockList{}
	err = c.r.Get().Namespace(c.ns).Resource("locks").LabelsSelectorParam(selector).Do().Into(result)
	return
}

// Get takes the name of the lock, and returns the corresponding Lock object, and an error if it occurs
func (c *locks) Get(name string) (result *api.Lock, err error) {
	result = &api.Lock{}
	err = c.r.Get().Namespace(c.ns).Resource("locks").Name(name).Do().Into(result)
	return
}

// Delete takes the name of the lock, and returns an error if one occurs
func (c *locks) Delete(name string) error {
	return c.r.Delete().Namespace(c.ns).Resource("locks").Name(name).Do().Error()
}

// Create takes the representation of a lock.  Returns the server's representation of the lock, and an error, if it occurs.
func (c *locks) Create(lock *api.Lock) (result *api.Lock, err error) {
	result = &api.Lock{}
	err = c.r.Post().Namespace(c.ns).Resource("locks").Body(lock).Do().Into(result)
	return
}

// Update takes the representation of a lock to update.  Returns the server's representation of the lock, and an error, if it occurs.
func (c *locks) Update(lock *api.Lock) (result *api.Lock, err error) {
	result = &api.Lock{}
	err = c.r.Put().Namespace(c.ns).Resource("locks").Name(lock.Name).Body(lock).Do().Into(result)
	return
}

// Watch returns a watch.Interface that watches the requested resource
func (c *locks) Watch(label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error) {
	return c.r.Get().
		Prefix("watch").
		Namespace(c.ns).
		Resource("locks").
		Param("resourceVersion", resourceVersion).
		LabelsSelectorParam(label).
		FieldsSelectorParam(field).
		Watch()
}
