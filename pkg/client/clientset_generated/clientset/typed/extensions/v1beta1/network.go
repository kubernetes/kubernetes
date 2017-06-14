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

package v1beta1

import (
	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	types "k8s.io/apimachinery/pkg/types"
	watch "k8s.io/apimachinery/pkg/watch"
	rest "k8s.io/client-go/rest"
	v1beta1 "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	scheme "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/scheme"
)

// NetworksGetter has a method to return a NetworkInterface.
// A group's client should implement this interface.
type NetworksGetter interface {
	Networks(namespace string) NetworkInterface
}

// NetworkInterface has methods to work with Network resources.
type NetworkInterface interface {
	Create(*v1beta1.Network) (*v1beta1.Network, error)
	Update(*v1beta1.Network) (*v1beta1.Network, error)
	Delete(name string, options *v1.DeleteOptions) error
	DeleteCollection(options *v1.DeleteOptions, listOptions v1.ListOptions) error
	Get(name string, options v1.GetOptions) (*v1beta1.Network, error)
	List(opts v1.ListOptions) (*v1beta1.NetworkList, error)
	Watch(opts v1.ListOptions) (watch.Interface, error)
	Patch(name string, pt types.PatchType, data []byte, subresources ...string) (result *v1beta1.Network, err error)
	NetworkExpansion
}

// networks implements NetworkInterface
type networks struct {
	client rest.Interface
	ns     string
}

// newNetworks returns a Networks
func newNetworks(c *ExtensionsV1beta1Client, namespace string) *networks {
	return &networks{
		client: c.RESTClient(),
		ns:     namespace,
	}
}

// Create takes the representation of a network and creates it.  Returns the server's representation of the network, and an error, if there is any.
func (c *networks) Create(network *v1beta1.Network) (result *v1beta1.Network, err error) {
	result = &v1beta1.Network{}
	err = c.client.Post().
		Namespace(c.ns).
		Resource("networks").
		Body(network).
		Do().
		Into(result)
	return
}

// Update takes the representation of a network and updates it. Returns the server's representation of the network, and an error, if there is any.
func (c *networks) Update(network *v1beta1.Network) (result *v1beta1.Network, err error) {
	result = &v1beta1.Network{}
	err = c.client.Put().
		Namespace(c.ns).
		Resource("networks").
		Name(network.Name).
		Body(network).
		Do().
		Into(result)
	return
}

// Delete takes name of the network and deletes it. Returns an error if one occurs.
func (c *networks) Delete(name string, options *v1.DeleteOptions) error {
	return c.client.Delete().
		Namespace(c.ns).
		Resource("networks").
		Name(name).
		Body(options).
		Do().
		Error()
}

// DeleteCollection deletes a collection of objects.
func (c *networks) DeleteCollection(options *v1.DeleteOptions, listOptions v1.ListOptions) error {
	return c.client.Delete().
		Namespace(c.ns).
		Resource("networks").
		VersionedParams(&listOptions, scheme.ParameterCodec).
		Body(options).
		Do().
		Error()
}

// Get takes name of the network, and returns the corresponding network object, and an error if there is any.
func (c *networks) Get(name string, options v1.GetOptions) (result *v1beta1.Network, err error) {
	result = &v1beta1.Network{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("networks").
		Name(name).
		VersionedParams(&options, scheme.ParameterCodec).
		Do().
		Into(result)
	return
}

// List takes label and field selectors, and returns the list of Networks that match those selectors.
func (c *networks) List(opts v1.ListOptions) (result *v1beta1.NetworkList, err error) {
	result = &v1beta1.NetworkList{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("networks").
		VersionedParams(&opts, scheme.ParameterCodec).
		Do().
		Into(result)
	return
}

// Watch returns a watch.Interface that watches the requested networks.
func (c *networks) Watch(opts v1.ListOptions) (watch.Interface, error) {
	opts.Watch = true
	return c.client.Get().
		Namespace(c.ns).
		Resource("networks").
		VersionedParams(&opts, scheme.ParameterCodec).
		Watch()
}

// Patch applies the patch and returns the patched network.
func (c *networks) Patch(name string, pt types.PatchType, data []byte, subresources ...string) (result *v1beta1.Network, err error) {
	result = &v1beta1.Network{}
	err = c.client.Patch(pt).
		Namespace(c.ns).
		Resource("networks").
		SubResource(subresources...).
		Name(name).
		Body(data).
		Do().
		Into(result)
	return
}
