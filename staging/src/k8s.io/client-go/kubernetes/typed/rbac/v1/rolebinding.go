/*
Copyright 2018 The Kubernetes Authors.

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

package v1

import (
	v1 "k8s.io/api/rbac/v1"
	meta_v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	types "k8s.io/apimachinery/pkg/types"
	watch "k8s.io/apimachinery/pkg/watch"
	scheme "k8s.io/client-go/kubernetes/scheme"
	rest "k8s.io/client-go/rest"
)

// RoleBindingsGetter has a method to return a RoleBindingInterface.
// A group's client should implement this interface.
type RoleBindingsGetter interface {
	RoleBindings(namespace string) RoleBindingInterface
}

// RoleBindingInterface has methods to work with RoleBinding resources.
type RoleBindingInterface interface {
	Create(*v1.RoleBinding) (*v1.RoleBinding, error)
	Update(*v1.RoleBinding) (*v1.RoleBinding, error)
	Delete(name string, options *meta_v1.DeleteOptions) error
	DeleteCollection(options *meta_v1.DeleteOptions, listOptions meta_v1.ListOptions) error
	Get(name string, options meta_v1.GetOptions) (*v1.RoleBinding, error)
	List(opts meta_v1.ListOptions) (*v1.RoleBindingList, error)
	Watch(opts meta_v1.ListOptions) (watch.Interface, error)
	Patch(name string, pt types.PatchType, data []byte, subresources ...string) (result *v1.RoleBinding, err error)
	RoleBindingExpansion
}

// roleBindings implements RoleBindingInterface
type roleBindings struct {
	client rest.Interface
	ns     string
}

// newRoleBindings returns a RoleBindings
func newRoleBindings(c *RbacV1Client, namespace string) *roleBindings {
	return &roleBindings{
		client: c.RESTClient(),
		ns:     namespace,
	}
}

// Get takes name of the roleBinding, and returns the corresponding roleBinding object, and an error if there is any.
func (c *roleBindings) Get(name string, options meta_v1.GetOptions) (result *v1.RoleBinding, err error) {
	result = &v1.RoleBinding{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("rolebindings").
		Name(name).
		VersionedParams(&options, scheme.ParameterCodec).
		Do().
		Into(result)
	return
}

// List takes label and field selectors, and returns the list of RoleBindings that match those selectors.
func (c *roleBindings) List(opts meta_v1.ListOptions) (result *v1.RoleBindingList, err error) {
	result = &v1.RoleBindingList{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("rolebindings").
		VersionedParams(&opts, scheme.ParameterCodec).
		Do().
		Into(result)
	return
}

// Watch returns a watch.Interface that watches the requested roleBindings.
func (c *roleBindings) Watch(opts meta_v1.ListOptions) (watch.Interface, error) {
	opts.Watch = true
	return c.client.Get().
		Namespace(c.ns).
		Resource("rolebindings").
		VersionedParams(&opts, scheme.ParameterCodec).
		Watch()
}

// Create takes the representation of a roleBinding and creates it.  Returns the server's representation of the roleBinding, and an error, if there is any.
func (c *roleBindings) Create(roleBinding *v1.RoleBinding) (result *v1.RoleBinding, err error) {
	result = &v1.RoleBinding{}
	err = c.client.Post().
		Namespace(c.ns).
		Resource("rolebindings").
		Body(roleBinding).
		Do().
		Into(result)
	return
}

// Update takes the representation of a roleBinding and updates it. Returns the server's representation of the roleBinding, and an error, if there is any.
func (c *roleBindings) Update(roleBinding *v1.RoleBinding) (result *v1.RoleBinding, err error) {
	result = &v1.RoleBinding{}
	err = c.client.Put().
		Namespace(c.ns).
		Resource("rolebindings").
		Name(roleBinding.Name).
		Body(roleBinding).
		Do().
		Into(result)
	return
}

// Delete takes name of the roleBinding and deletes it. Returns an error if one occurs.
func (c *roleBindings) Delete(name string, options *meta_v1.DeleteOptions) error {
	return c.client.Delete().
		Namespace(c.ns).
		Resource("rolebindings").
		Name(name).
		Body(options).
		Do().
		Error()
}

// DeleteCollection deletes a collection of objects.
func (c *roleBindings) DeleteCollection(options *meta_v1.DeleteOptions, listOptions meta_v1.ListOptions) error {
	return c.client.Delete().
		Namespace(c.ns).
		Resource("rolebindings").
		VersionedParams(&listOptions, scheme.ParameterCodec).
		Body(options).
		Do().
		Error()
}

// Patch applies the patch and returns the patched roleBinding.
func (c *roleBindings) Patch(name string, pt types.PatchType, data []byte, subresources ...string) (result *v1.RoleBinding, err error) {
	result = &v1.RoleBinding{}
	err = c.client.Patch(pt).
		Namespace(c.ns).
		Resource("rolebindings").
		SubResource(subresources...).
		Name(name).
		Body(data).
		Do().
		Into(result)
	return
}
