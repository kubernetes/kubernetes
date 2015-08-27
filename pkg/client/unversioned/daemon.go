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
	"k8s.io/kubernetes/pkg/expapi"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/watch"
)

// DaemonsNamespacer has methods to work with Daemon resources in a namespace
type DaemonsNamespacer interface {
	Daemons(namespace string) DaemonInterface
}

type DaemonInterface interface {
	List(selector labels.Selector) (*expapi.DaemonList, error)
	Get(name string) (*expapi.Daemon, error)
	Create(ctrl *expapi.Daemon) (*expapi.Daemon, error)
	Update(ctrl *expapi.Daemon) (*expapi.Daemon, error)
	Delete(name string) error
	Watch(label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error)
}

// daemons implements DaemonsNamespacer interface
type daemons struct {
	r  *ExperimentalClient
	ns string
}

func newDaemons(c *ExperimentalClient, namespace string) *daemons {
	return &daemons{c, namespace}
}

// Ensure statically that daemons implements DaemonInterface.
var _ DaemonInterface = &daemons{}

func (c *daemons) List(selector labels.Selector) (result *expapi.DaemonList, err error) {
	result = &expapi.DaemonList{}
	err = c.r.Get().Namespace(c.ns).Resource("daemons").LabelsSelectorParam(selector).Do().Into(result)
	return
}

// Get returns information about a particular daemon.
func (c *daemons) Get(name string) (result *expapi.Daemon, err error) {
	result = &expapi.Daemon{}
	err = c.r.Get().Namespace(c.ns).Resource("daemons").Name(name).Do().Into(result)
	return
}

// Create creates a new daemon.
func (c *daemons) Create(daemon *expapi.Daemon) (result *expapi.Daemon, err error) {
	result = &expapi.Daemon{}
	err = c.r.Post().Namespace(c.ns).Resource("daemons").Body(daemon).Do().Into(result)
	return
}

// Update updates an existing daemon.
func (c *daemons) Update(daemon *expapi.Daemon) (result *expapi.Daemon, err error) {
	result = &expapi.Daemon{}
	err = c.r.Put().Namespace(c.ns).Resource("daemons").Name(daemon.Name).Body(daemon).Do().Into(result)
	return
}

// Delete deletes an existing daemon.
func (c *daemons) Delete(name string) error {
	return c.r.Delete().Namespace(c.ns).Resource("daemons").Name(name).Do().Error()
}

// Watch returns a watch.Interface that watches the requested daemons.
func (c *daemons) Watch(label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error) {
	return c.r.Get().
		Prefix("watch").
		Namespace(c.ns).
		Resource("daemons").
		Param("resourceVersion", resourceVersion).
		LabelsSelectorParam(label).
		FieldsSelectorParam(field).
		Watch()
}
