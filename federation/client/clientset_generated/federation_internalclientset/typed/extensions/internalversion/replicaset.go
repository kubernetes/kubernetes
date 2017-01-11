/*
Copyright 2017 The Kubernetes Authors.

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

package internalversion

import (
	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	watch "k8s.io/apimachinery/pkg/watch"
	api "k8s.io/kubernetes/pkg/api"
	extensions "k8s.io/kubernetes/pkg/apis/extensions"
	restclient "k8s.io/kubernetes/pkg/client/restclient"
)

// ReplicaSetsGetter has a method to return a ReplicaSetInterface.
// A group's client should implement this interface.
type ReplicaSetsGetter interface {
	ReplicaSets(namespace string) ReplicaSetInterface
}

// ReplicaSetInterface has methods to work with ReplicaSet resources.
type ReplicaSetInterface interface {
	Create(*extensions.ReplicaSet) (*extensions.ReplicaSet, error)
	Update(*extensions.ReplicaSet) (*extensions.ReplicaSet, error)
	UpdateStatus(*extensions.ReplicaSet) (*extensions.ReplicaSet, error)
	Delete(name string, options *api.DeleteOptions) error
	DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error
	Get(name string, options v1.GetOptions) (*extensions.ReplicaSet, error)
	List(opts api.ListOptions) (*extensions.ReplicaSetList, error)
	Watch(opts api.ListOptions) (watch.Interface, error)
	Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *extensions.ReplicaSet, err error)
	ReplicaSetExpansion
}

// replicaSets implements ReplicaSetInterface
type replicaSets struct {
	client restclient.Interface
	ns     string
}

// newReplicaSets returns a ReplicaSets
func newReplicaSets(c *ExtensionsClient, namespace string) *replicaSets {
	return &replicaSets{
		client: c.RESTClient(),
		ns:     namespace,
	}
}

// Create takes the representation of a replicaSet and creates it.  Returns the server's representation of the replicaSet, and an error, if there is any.
func (c *replicaSets) Create(replicaSet *extensions.ReplicaSet) (result *extensions.ReplicaSet, err error) {
	result = &extensions.ReplicaSet{}
	err = c.client.Post().
		Namespace(c.ns).
		Resource("replicasets").
		Body(replicaSet).
		Do().
		Into(result)
	return
}

// Update takes the representation of a replicaSet and updates it. Returns the server's representation of the replicaSet, and an error, if there is any.
func (c *replicaSets) Update(replicaSet *extensions.ReplicaSet) (result *extensions.ReplicaSet, err error) {
	result = &extensions.ReplicaSet{}
	err = c.client.Put().
		Namespace(c.ns).
		Resource("replicasets").
		Name(replicaSet.Name).
		Body(replicaSet).
		Do().
		Into(result)
	return
}

// UpdateStatus was generated because the type contains a Status member.
// Add a +genclientstatus=false comment above the type to avoid generating UpdateStatus().

func (c *replicaSets) UpdateStatus(replicaSet *extensions.ReplicaSet) (result *extensions.ReplicaSet, err error) {
	result = &extensions.ReplicaSet{}
	err = c.client.Put().
		Namespace(c.ns).
		Resource("replicasets").
		Name(replicaSet.Name).
		SubResource("status").
		Body(replicaSet).
		Do().
		Into(result)
	return
}

// Delete takes name of the replicaSet and deletes it. Returns an error if one occurs.
func (c *replicaSets) Delete(name string, options *api.DeleteOptions) error {
	return c.client.Delete().
		Namespace(c.ns).
		Resource("replicasets").
		Name(name).
		Body(options).
		Do().
		Error()
}

// DeleteCollection deletes a collection of objects.
func (c *replicaSets) DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error {
	return c.client.Delete().
		Namespace(c.ns).
		Resource("replicasets").
		VersionedParams(&listOptions, api.ParameterCodec).
		Body(options).
		Do().
		Error()
}

// Get takes name of the replicaSet, and returns the corresponding replicaSet object, and an error if there is any.
func (c *replicaSets) Get(name string, options v1.GetOptions) (result *extensions.ReplicaSet, err error) {
	result = &extensions.ReplicaSet{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("replicasets").
		Name(name).
		VersionedParams(&options, api.ParameterCodec).
		Do().
		Into(result)
	return
}

// List takes label and field selectors, and returns the list of ReplicaSets that match those selectors.
func (c *replicaSets) List(opts api.ListOptions) (result *extensions.ReplicaSetList, err error) {
	result = &extensions.ReplicaSetList{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("replicasets").
		VersionedParams(&opts, api.ParameterCodec).
		Do().
		Into(result)
	return
}

// Watch returns a watch.Interface that watches the requested replicaSets.
func (c *replicaSets) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.client.Get().
		Prefix("watch").
		Namespace(c.ns).
		Resource("replicasets").
		VersionedParams(&opts, api.ParameterCodec).
		Watch()
}

// Patch applies the patch and returns the patched replicaSet.
func (c *replicaSets) Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *extensions.ReplicaSet, err error) {
	result = &extensions.ReplicaSet{}
	err = c.client.Patch(pt).
		Namespace(c.ns).
		Resource("replicasets").
		SubResource(subresources...).
		Name(name).
		Body(data).
		Do().
		Into(result)
	return
}
