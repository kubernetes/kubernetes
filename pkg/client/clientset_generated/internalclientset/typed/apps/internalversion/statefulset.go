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
	apps "k8s.io/kubernetes/pkg/apis/apps"
	restclient "k8s.io/kubernetes/pkg/client/restclient"
)

// StatefulSetsGetter has a method to return a StatefulSetInterface.
// A group's client should implement this interface.
type StatefulSetsGetter interface {
	StatefulSets(namespace string) StatefulSetInterface
}

// StatefulSetInterface has methods to work with StatefulSet resources.
type StatefulSetInterface interface {
	Create(*apps.StatefulSet) (*apps.StatefulSet, error)
	Update(*apps.StatefulSet) (*apps.StatefulSet, error)
	UpdateStatus(*apps.StatefulSet) (*apps.StatefulSet, error)
	Delete(name string, options *api.DeleteOptions) error
	DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error
	Get(name string, options v1.GetOptions) (*apps.StatefulSet, error)
	List(opts api.ListOptions) (*apps.StatefulSetList, error)
	Watch(opts api.ListOptions) (watch.Interface, error)
	Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *apps.StatefulSet, err error)
	StatefulSetExpansion
}

// statefulSets implements StatefulSetInterface
type statefulSets struct {
	client restclient.Interface
	ns     string
}

// newStatefulSets returns a StatefulSets
func newStatefulSets(c *AppsClient, namespace string) *statefulSets {
	return &statefulSets{
		client: c.RESTClient(),
		ns:     namespace,
	}
}

// Create takes the representation of a statefulSet and creates it.  Returns the server's representation of the statefulSet, and an error, if there is any.
func (c *statefulSets) Create(statefulSet *apps.StatefulSet) (result *apps.StatefulSet, err error) {
	result = &apps.StatefulSet{}
	err = c.client.Post().
		Namespace(c.ns).
		Resource("statefulsets").
		Body(statefulSet).
		Do().
		Into(result)
	return
}

// Update takes the representation of a statefulSet and updates it. Returns the server's representation of the statefulSet, and an error, if there is any.
func (c *statefulSets) Update(statefulSet *apps.StatefulSet) (result *apps.StatefulSet, err error) {
	result = &apps.StatefulSet{}
	err = c.client.Put().
		Namespace(c.ns).
		Resource("statefulsets").
		Name(statefulSet.Name).
		Body(statefulSet).
		Do().
		Into(result)
	return
}

// UpdateStatus was generated because the type contains a Status member.
// Add a +genclientstatus=false comment above the type to avoid generating UpdateStatus().

func (c *statefulSets) UpdateStatus(statefulSet *apps.StatefulSet) (result *apps.StatefulSet, err error) {
	result = &apps.StatefulSet{}
	err = c.client.Put().
		Namespace(c.ns).
		Resource("statefulsets").
		Name(statefulSet.Name).
		SubResource("status").
		Body(statefulSet).
		Do().
		Into(result)
	return
}

// Delete takes name of the statefulSet and deletes it. Returns an error if one occurs.
func (c *statefulSets) Delete(name string, options *api.DeleteOptions) error {
	return c.client.Delete().
		Namespace(c.ns).
		Resource("statefulsets").
		Name(name).
		Body(options).
		Do().
		Error()
}

// DeleteCollection deletes a collection of objects.
func (c *statefulSets) DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error {
	return c.client.Delete().
		Namespace(c.ns).
		Resource("statefulsets").
		VersionedParams(&listOptions, api.ParameterCodec).
		Body(options).
		Do().
		Error()
}

// Get takes name of the statefulSet, and returns the corresponding statefulSet object, and an error if there is any.
func (c *statefulSets) Get(name string, options v1.GetOptions) (result *apps.StatefulSet, err error) {
	result = &apps.StatefulSet{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("statefulsets").
		Name(name).
		VersionedParams(&options, api.ParameterCodec).
		Do().
		Into(result)
	return
}

// List takes label and field selectors, and returns the list of StatefulSets that match those selectors.
func (c *statefulSets) List(opts api.ListOptions) (result *apps.StatefulSetList, err error) {
	result = &apps.StatefulSetList{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("statefulsets").
		VersionedParams(&opts, api.ParameterCodec).
		Do().
		Into(result)
	return
}

// Watch returns a watch.Interface that watches the requested statefulSets.
func (c *statefulSets) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.client.Get().
		Prefix("watch").
		Namespace(c.ns).
		Resource("statefulsets").
		VersionedParams(&opts, api.ParameterCodec).
		Watch()
}

// Patch applies the patch and returns the patched statefulSet.
func (c *statefulSets) Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *apps.StatefulSet, err error) {
	result = &apps.StatefulSet{}
	err = c.client.Patch(pt).
		Namespace(c.ns).
		Resource("statefulsets").
		SubResource(subresources...).
		Name(name).
		Body(data).
		Do().
		Into(result)
	return
}
