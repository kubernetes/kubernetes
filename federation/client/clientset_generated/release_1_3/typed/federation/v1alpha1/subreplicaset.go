/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package v1alpha1

import (
	v1alpha1 "k8s.io/kubernetes/federation/apis/federation/v1alpha1"
	api "k8s.io/kubernetes/pkg/api"
	watch "k8s.io/kubernetes/pkg/watch"
)

// SubReplicaSetsGetter has a method to return a SubReplicaSetInterface.
// A group's client should implement this interface.
type SubReplicaSetsGetter interface {
	SubReplicaSets(namespace string) SubReplicaSetInterface
}

// SubReplicaSetInterface has methods to work with SubReplicaSet resources.
type SubReplicaSetInterface interface {
	Create(*v1alpha1.SubReplicaSet) (*v1alpha1.SubReplicaSet, error)
	Update(*v1alpha1.SubReplicaSet) (*v1alpha1.SubReplicaSet, error)
	UpdateStatus(*v1alpha1.SubReplicaSet) (*v1alpha1.SubReplicaSet, error)
	Delete(name string, options *api.DeleteOptions) error
	DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error
	Get(name string) (*v1alpha1.SubReplicaSet, error)
	List(opts api.ListOptions) (*v1alpha1.SubReplicaSetList, error)
	Watch(opts api.ListOptions) (watch.Interface, error)
	SubReplicaSetExpansion
}

// subReplicaSets implements SubReplicaSetInterface
type subReplicaSets struct {
	client *FederationClient
	ns     string
}

// newSubReplicaSets returns a SubReplicaSets
func newSubReplicaSets(c *FederationClient, namespace string) *subReplicaSets {
	return &subReplicaSets{
		client: c,
		ns:     namespace,
	}
}

// Create takes the representation of a subReplicaSet and creates it.  Returns the server's representation of the subReplicaSet, and an error, if there is any.
func (c *subReplicaSets) Create(subReplicaSet *v1alpha1.SubReplicaSet) (result *v1alpha1.SubReplicaSet, err error) {
	result = &v1alpha1.SubReplicaSet{}
	err = c.client.Post().
		Namespace(c.ns).
		Resource("subreplicasets").
		Body(subReplicaSet).
		Do().
		Into(result)
	return
}

// Update takes the representation of a subReplicaSet and updates it. Returns the server's representation of the subReplicaSet, and an error, if there is any.
func (c *subReplicaSets) Update(subReplicaSet *v1alpha1.SubReplicaSet) (result *v1alpha1.SubReplicaSet, err error) {
	result = &v1alpha1.SubReplicaSet{}
	err = c.client.Put().
		Namespace(c.ns).
		Resource("subreplicasets").
		Name(subReplicaSet.Name).
		Body(subReplicaSet).
		Do().
		Into(result)
	return
}

func (c *subReplicaSets) UpdateStatus(subReplicaSet *v1alpha1.SubReplicaSet) (result *v1alpha1.SubReplicaSet, err error) {
	result = &v1alpha1.SubReplicaSet{}
	err = c.client.Put().
		Namespace(c.ns).
		Resource("subreplicasets").
		Name(subReplicaSet.Name).
		SubResource("status").
		Body(subReplicaSet).
		Do().
		Into(result)
	return
}

// Delete takes name of the subReplicaSet and deletes it. Returns an error if one occurs.
func (c *subReplicaSets) Delete(name string, options *api.DeleteOptions) error {
	return c.client.Delete().
		Namespace(c.ns).
		Resource("subreplicasets").
		Name(name).
		Body(options).
		Do().
		Error()
}

// DeleteCollection deletes a collection of objects.
func (c *subReplicaSets) DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error {
	return c.client.Delete().
		Namespace(c.ns).
		Resource("subreplicasets").
		VersionedParams(&listOptions, api.ParameterCodec).
		Body(options).
		Do().
		Error()
}

// Get takes name of the subReplicaSet, and returns the corresponding subReplicaSet object, and an error if there is any.
func (c *subReplicaSets) Get(name string) (result *v1alpha1.SubReplicaSet, err error) {
	result = &v1alpha1.SubReplicaSet{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("subreplicasets").
		Name(name).
		Do().
		Into(result)
	return
}

// List takes label and field selectors, and returns the list of SubReplicaSets that match those selectors.
func (c *subReplicaSets) List(opts api.ListOptions) (result *v1alpha1.SubReplicaSetList, err error) {
	result = &v1alpha1.SubReplicaSetList{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("subreplicasets").
		VersionedParams(&opts, api.ParameterCodec).
		Do().
		Into(result)
	return
}

// Watch returns a watch.Interface that watches the requested subReplicaSets.
func (c *subReplicaSets) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.client.Get().
		Prefix("watch").
		Namespace(c.ns).
		Resource("subreplicasets").
		VersionedParams(&opts, api.ParameterCodec).
		Watch()
}
