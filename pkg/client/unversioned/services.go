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
	"k8s.io/kubernetes/pkg/client/restclient"
	"k8s.io/kubernetes/pkg/util/net"
	"k8s.io/kubernetes/pkg/watch"
)

// ServicesNamespacer has methods to work with Service resources in a namespace
type ServicesNamespacer interface {
	Services(namespace string) ServiceInterface
}

// ServiceInterface has methods to work with Service resources.
type ServiceInterface interface {
	List(opts api.ListOptions) (*api.ServiceList, error)
	Get(name string) (*api.Service, error)
	Create(srv *api.Service) (*api.Service, error)
	Update(srv *api.Service) (*api.Service, error)
	UpdateStatus(srv *api.Service) (*api.Service, error)
	Delete(name string) error
	Watch(opts api.ListOptions) (watch.Interface, error)
	ProxyGet(scheme, name, port, path string, params map[string]string) restclient.ResponseWrapper
}

// services implements ServicesNamespacer interface
type services struct {
	r  *Client
	ns string
}

// newServices returns a services
func newServices(c *Client, namespace string) *services {
	return &services{c, namespace}
}

// List takes a selector, and returns the list of services that match that selector
func (c *services) List(opts api.ListOptions) (result *api.ServiceList, err error) {
	result = &api.ServiceList{}
	err = c.r.Get().
		Namespace(c.ns).
		Resource("services").
		VersionedParams(&opts, api.ParameterCodec).
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

// UpdateStatus takes a Service object with the new status and applies it as an update to the existing Service.
func (c *services) UpdateStatus(service *api.Service) (result *api.Service, err error) {
	result = &api.Service{}
	err = c.r.Put().Namespace(c.ns).Resource("services").Name(service.Name).SubResource("status").Body(service).Do().Into(result)
	return
}

// Delete deletes an existing service.
func (c *services) Delete(name string) error {
	return c.r.Delete().Namespace(c.ns).Resource("services").Name(name).Do().Error()
}

// Watch returns a watch.Interface that watches the requested services.
func (c *services) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.r.Get().
		Prefix("watch").
		Namespace(c.ns).
		Resource("services").
		VersionedParams(&opts, api.ParameterCodec).
		Watch()
}

// ProxyGet returns a response of the service by calling it through the proxy.
func (c *services) ProxyGet(scheme, name, port, path string, params map[string]string) restclient.ResponseWrapper {
	request := c.r.Get().
		Namespace(c.ns).
		Resource("services").
		SubResource("proxy").
		Name(net.JoinSchemeNamePort(scheme, name, port)).
		Suffix(path)
	for k, v := range params {
		request = request.Param(k, v)
	}
	return request
}
