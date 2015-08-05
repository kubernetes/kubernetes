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

package client

import (
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/watch"
)

// ServicesNamespacer has methods to work with Service resources in a namespace
type ServicesNamespacer interface {
	Services(namespace string) ServiceInterface
}

// ServiceInterface has methods to work with Service resources.
type ServiceInterface interface {
	List(selector labels.Selector) (*api.ServiceList, error)
	Get(name string) (*api.Service, error)
	Create(srv *api.Service) (*api.Service, error)
	Update(srv *api.Service) (*api.Service, error)
	Delete(name string) error
	Watch(label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error)
}

// services implements PodsNamespacer interface
type services struct {
	r  *Client
	ns string
}

// newServices returns a PodsClient
func newServices(c *Client, namespace string) *services {
	return &services{c, namespace}
}

// List takes a selector, and returns the list of services that match that selector
func (c *services) List(selector labels.Selector) (result *api.ServiceList, err error) {
	result = &api.ServiceList{}
	err = c.r.Get().
		Namespace(c.ns).
		Resource("services").
		LabelsSelectorParam(selector).
		Do().
		Into(result)
	return
}

// Get returns information about a particular service.
func (c *services) Get(name string) (result *api.Service, err error) {
	result = &api.Service{}
	err = c.r.Get().Namespace(c.ns).Resource("services").Name(name).Do().Into(result)
	return
}

// Create creates a new service.
func (c *services) Create(svc *api.Service) (result *api.Service, err error) {
	result = &api.Service{}
	err = c.r.Post().Namespace(c.ns).Resource("services").Body(svc).Do().Into(result)
	return
}

// Update updates an existing service.
func (c *services) Update(svc *api.Service) (result *api.Service, err error) {
	result = &api.Service{}
	err = c.r.Put().Namespace(c.ns).Resource("services").Name(svc.Name).Body(svc).Do().Into(result)
	return
}

// Delete deletes an existing service.
func (c *services) Delete(name string) error {
	return c.r.Delete().Namespace(c.ns).Resource("services").Name(name).Do().Error()
}

// Watch returns a watch.Interface that watches the requested services.
func (c *services) Watch(label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error) {
	return c.r.Get().
		Prefix("watch").
		Namespace(c.ns).
		Resource("services").
		Param("resourceVersion", resourceVersion).
		LabelsSelectorParam(label).
		FieldsSelectorParam(field).
		Watch()
}
