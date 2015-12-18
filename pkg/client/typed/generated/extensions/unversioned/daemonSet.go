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
	extensions "k8s.io/kubernetes/pkg/apis/extensions"
	watch "k8s.io/kubernetes/pkg/watch"
)

// DaemonSetNamespacer has methods to work with DaemonSet resources in a namespace
type DaemonSetNamespacer interface {
	DaemonSets(namespace string) DaemonSetInterface
}

// DaemonSetInterface has methods to work with DaemonSet resources.
type DaemonSetInterface interface {
	Create(*extensions.DaemonSet) (*extensions.DaemonSet, error)
	Update(*extensions.DaemonSet) (*extensions.DaemonSet, error)
	Delete(name string, options *api.DeleteOptions) error
	Get(name string) (*extensions.DaemonSet, error)
	List(opts unversioned.ListOptions) (*extensions.DaemonSetList, error)
	Watch(opts unversioned.ListOptions) (watch.Interface, error)
}

// daemonSets implements DaemonSetInterface
type daemonSets struct {
	client *ExtensionsClient
	ns     string
}

// newDaemonSets returns a DaemonSets
func newDaemonSets(c *ExtensionsClient, namespace string) *daemonSets {
	return &daemonSets{
		client: c,
		ns:     namespace,
	}
}

// Create takes the representation of a daemonSet and creates it.  Returns the server's representation of the daemonSet, and an error, if there is any.
func (c *daemonSets) Create(daemonSet *extensions.DaemonSet) (result *extensions.DaemonSet, err error) {
	result = &extensions.DaemonSet{}
	err = c.client.Post().
		Namespace(c.ns).
		Resource("daemonSets").
		Body(daemonSet).
		Do().
		Into(result)
	return
}

// Update takes the representation of a daemonSet and updates it. Returns the server's representation of the daemonSet, and an error, if there is any.
func (c *daemonSets) Update(daemonSet *extensions.DaemonSet) (result *extensions.DaemonSet, err error) {
	result = &extensions.DaemonSet{}
	err = c.client.Put().
		Namespace(c.ns).
		Resource("daemonSets").
		Name(daemonSet.Name).
		Body(daemonSet).
		Do().
		Into(result)
	return
}

// Delete takes name of the daemonSet and deletes it. Returns an error if one occurs.
func (c *daemonSets) Delete(name string, options *api.DeleteOptions) error {
	if options == nil {
		return c.client.Delete().Namespace(c.ns).Resource("daemonSets").Name(name).Do().Error()
	}
	body, err := api.Scheme.EncodeToVersion(options, c.client.APIVersion().String())
	if err != nil {
		return err
	}
	return c.client.Delete().
		Namespace(c.ns).
		Resource("daemonSets").
		Name(name).
		Body(body).
		Do().
		Error()
}

// Get takes name of the daemonSet, and returns the corresponding daemonSet object, and an error if there is any.
func (c *daemonSets) Get(name string) (result *extensions.DaemonSet, err error) {
	result = &extensions.DaemonSet{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("daemonSets").
		Name(name).
		Do().
		Into(result)
	return
}

// List takes label and field selectors, and returns the list of DaemonSets that match those selectors.
func (c *daemonSets) List(opts unversioned.ListOptions) (result *extensions.DaemonSetList, err error) {
	result = &extensions.DaemonSetList{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("daemonSets").
		VersionedParams(&opts, api.Scheme).
		Do().
		Into(result)
	return
}

// Watch returns a watch.Interface that watches the requested daemonSets.
func (c *daemonSets) Watch(opts unversioned.ListOptions) (watch.Interface, error) {
	return c.client.Get().
		Prefix("watch").
		Namespace(c.ns).
		Resource("daemonSets").
		VersionedParams(&opts, api.Scheme).
		Watch()
}
