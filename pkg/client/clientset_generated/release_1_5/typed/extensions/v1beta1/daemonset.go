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

package v1beta1

import (
	api "k8s.io/kubernetes/pkg/api"
	v1 "k8s.io/kubernetes/pkg/api/v1"
	v1beta1 "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	restclient "k8s.io/kubernetes/pkg/client/restclient"
	watch "k8s.io/kubernetes/pkg/watch"
)

// DaemonSetsGetter has a method to return a DaemonSetInterface.
// A group's client should implement this interface.
type DaemonSetsGetter interface {
	DaemonSets(namespace string) DaemonSetInterface
}

// DaemonSetInterface has methods to work with DaemonSet resources.
type DaemonSetInterface interface {
	Create(*v1beta1.DaemonSet) (*v1beta1.DaemonSet, error)
	Update(*v1beta1.DaemonSet) (*v1beta1.DaemonSet, error)
	UpdateStatus(*v1beta1.DaemonSet) (*v1beta1.DaemonSet, error)
	Delete(name string, options *v1.DeleteOptions) error
	DeleteCollection(options *v1.DeleteOptions, listOptions v1.ListOptions) error
	Get(name string) (*v1beta1.DaemonSet, error)
	List(opts v1.ListOptions) (*v1beta1.DaemonSetList, error)
	Watch(opts v1.ListOptions) (watch.Interface, error)
	Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *v1beta1.DaemonSet, err error)
	DaemonSetExpansion
}

// daemonSets implements DaemonSetInterface
type daemonSets struct {
	client restclient.Interface
	ns     string
}

// newDaemonSets returns a DaemonSets
func newDaemonSets(c *ExtensionsV1beta1Client, namespace string) *daemonSets {
	return &daemonSets{
		client: c.RESTClient(),
		ns:     namespace,
	}
}

// Create takes the representation of a daemonSet and creates it.  Returns the server's representation of the daemonSet, and an error, if there is any.
func (c *daemonSets) Create(daemonSet *v1beta1.DaemonSet) (result *v1beta1.DaemonSet, err error) {
	result = &v1beta1.DaemonSet{}
	err = c.client.Post().
		Namespace(c.ns).
		Resource("daemonsets").
		Body(daemonSet).
		Do().
		Into(result)
	return
}

// Update takes the representation of a daemonSet and updates it. Returns the server's representation of the daemonSet, and an error, if there is any.
func (c *daemonSets) Update(daemonSet *v1beta1.DaemonSet) (result *v1beta1.DaemonSet, err error) {
	result = &v1beta1.DaemonSet{}
	err = c.client.Put().
		Namespace(c.ns).
		Resource("daemonsets").
		Name(daemonSet.Name).
		Body(daemonSet).
		Do().
		Into(result)
	return
}

func (c *daemonSets) UpdateStatus(daemonSet *v1beta1.DaemonSet) (result *v1beta1.DaemonSet, err error) {
	result = &v1beta1.DaemonSet{}
	err = c.client.Put().
		Namespace(c.ns).
		Resource("daemonsets").
		Name(daemonSet.Name).
		SubResource("status").
		Body(daemonSet).
		Do().
		Into(result)
	return
}

// Delete takes name of the daemonSet and deletes it. Returns an error if one occurs.
func (c *daemonSets) Delete(name string, options *v1.DeleteOptions) error {
	return c.client.Delete().
		Namespace(c.ns).
		Resource("daemonsets").
		Name(name).
		Body(options).
		Do().
		Error()
}

// DeleteCollection deletes a collection of objects.
func (c *daemonSets) DeleteCollection(options *v1.DeleteOptions, listOptions v1.ListOptions) error {
	return c.client.Delete().
		Namespace(c.ns).
		Resource("daemonsets").
		VersionedParams(&listOptions, api.ParameterCodec).
		Body(options).
		Do().
		Error()
}

// Get takes name of the daemonSet, and returns the corresponding daemonSet object, and an error if there is any.
func (c *daemonSets) Get(name string) (result *v1beta1.DaemonSet, err error) {
	result = &v1beta1.DaemonSet{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("daemonsets").
		Name(name).
		Do().
		Into(result)
	return
}

// List takes label and field selectors, and returns the list of DaemonSets that match those selectors.
func (c *daemonSets) List(opts v1.ListOptions) (result *v1beta1.DaemonSetList, err error) {
	result = &v1beta1.DaemonSetList{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("daemonsets").
		VersionedParams(&opts, api.ParameterCodec).
		Do().
		Into(result)
	return
}

// Watch returns a watch.Interface that watches the requested daemonSets.
func (c *daemonSets) Watch(opts v1.ListOptions) (watch.Interface, error) {
	return c.client.Get().
		Prefix("watch").
		Namespace(c.ns).
		Resource("daemonsets").
		VersionedParams(&opts, api.ParameterCodec).
		Watch()
}

// Patch applies the patch and returns the patched daemonSet.
func (c *daemonSets) Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *v1beta1.DaemonSet, err error) {
	result = &v1beta1.DaemonSet{}
	err = c.client.Patch(pt).
		Namespace(c.ns).
		Resource("daemonsets").
		SubResource(subresources...).
		Name(name).
		Body(data).
		Do().
		Into(result)
	return
}
