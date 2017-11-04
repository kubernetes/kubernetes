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

package v1beta2

import (
	v1beta2 "k8s.io/api/apps/v1beta2"
	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	types "k8s.io/apimachinery/pkg/types"
	watch "k8s.io/apimachinery/pkg/watch"
	scheme "k8s.io/client-go/kubernetes/scheme"
	rest "k8s.io/client-go/rest"
)

// ReplicaSetsGetter has a method to return a ReplicaSetInterface.
// A group's client should implement this interface.
type ReplicaSetsGetter interface {
	ReplicaSets(namespace string) ReplicaSetInterface
}

// ReplicaSetInterface has methods to work with ReplicaSet resources.
type ReplicaSetInterface interface {
	Create(*v1beta2.ReplicaSet) (*v1beta2.ReplicaSet, error)
	Update(*v1beta2.ReplicaSet) (*v1beta2.ReplicaSet, error)
	UpdateStatus(*v1beta2.ReplicaSet) (*v1beta2.ReplicaSet, error)
	Delete(name string, options *v1.DeleteOptions) error
	DeleteCollection(options *v1.DeleteOptions, listOptions v1.ListOptions) error
	Get(name string, options v1.GetOptions) (*v1beta2.ReplicaSet, error)
	List(opts v1.ListOptions) (*v1beta2.ReplicaSetList, error)
	Watch(opts v1.ListOptions) (watch.Interface, error)
	Patch(name string, pt types.PatchType, data []byte, subresources ...string) (result *v1beta2.ReplicaSet, err error)
	ReplicaSetExpansion
}

// replicaSets implements ReplicaSetInterface
type replicaSets struct {
	client rest.Interface
	ns     string
}

// newReplicaSets returns a ReplicaSets
func newReplicaSets(c *AppsV1beta2Client, namespace string) *replicaSets {
	return &replicaSets{
		client: c.RESTClient(),
		ns:     namespace,
	}
}

// Get takes name of the replicaSet, and returns the corresponding replicaSet object, and an error if there is any.
func (c *replicaSets) Get(name string, options v1.GetOptions) (result *v1beta2.ReplicaSet, err error) {
	result = &v1beta2.ReplicaSet{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("replicasets").
		Name(name).
		VersionedParams(&options, scheme.ParameterCodec).
		Do().
		Into(result)
	return
}

// List takes label and field selectors, and returns the list of ReplicaSets that match those selectors.
func (c *replicaSets) List(opts v1.ListOptions) (result *v1beta2.ReplicaSetList, err error) {
	result = &v1beta2.ReplicaSetList{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("replicasets").
		VersionedParams(&opts, scheme.ParameterCodec).
		Do().
		Into(result)
	return
}

// Watch returns a watch.Interface that watches the requested replicaSets.
func (c *replicaSets) Watch(opts v1.ListOptions) (watch.Interface, error) {
	opts.Watch = true
	return c.client.Get().
		Namespace(c.ns).
		Resource("replicasets").
		VersionedParams(&opts, scheme.ParameterCodec).
		Watch()
}

// Create takes the representation of a replicaSet and creates it.  Returns the server's representation of the replicaSet, and an error, if there is any.
func (c *replicaSets) Create(replicaSet *v1beta2.ReplicaSet) (result *v1beta2.ReplicaSet, err error) {
	result = &v1beta2.ReplicaSet{}
	err = c.client.Post().
		Namespace(c.ns).
		Resource("replicasets").
		Body(replicaSet).
		Do().
		Into(result)
	return
}

// Update takes the representation of a replicaSet and updates it. Returns the server's representation of the replicaSet, and an error, if there is any.
func (c *replicaSets) Update(replicaSet *v1beta2.ReplicaSet) (result *v1beta2.ReplicaSet, err error) {
	result = &v1beta2.ReplicaSet{}
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
// Add a +genclient:noStatus comment above the type to avoid generating UpdateStatus().

func (c *replicaSets) UpdateStatus(replicaSet *v1beta2.ReplicaSet) (result *v1beta2.ReplicaSet, err error) {
	result = &v1beta2.ReplicaSet{}
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
func (c *replicaSets) Delete(name string, options *v1.DeleteOptions) error {
	return c.client.Delete().
		Namespace(c.ns).
		Resource("replicasets").
		Name(name).
		Body(options).
		Do().
		Error()
}

// DeleteCollection deletes a collection of objects.
func (c *replicaSets) DeleteCollection(options *v1.DeleteOptions, listOptions v1.ListOptions) error {
	return c.client.Delete().
		Namespace(c.ns).
		Resource("replicasets").
		VersionedParams(&listOptions, scheme.ParameterCodec).
		Body(options).
		Do().
		Error()
}

// Patch applies the patch and returns the patched replicaSet.
func (c *replicaSets) Patch(name string, pt types.PatchType, data []byte, subresources ...string) (result *v1beta2.ReplicaSet, err error) {
	result = &v1beta2.ReplicaSet{}
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
