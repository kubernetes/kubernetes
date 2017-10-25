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

package v1alpha1

import (
	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	types "k8s.io/apimachinery/pkg/types"
	watch "k8s.io/apimachinery/pkg/watch"
	rest "k8s.io/client-go/rest"
	scheme "k8s.io/kubernetes/pkg/controller/node/nodeconfig/client-go/clientset/versioned/scheme"
	v1alpha1 "k8s.io/kubernetes/pkg/controller/node/nodeconfig/v1alpha1"

	"github.com/golang/glog"
)

// NodeConfigSourcePoolsGetter has a method to return a NodeConfigSourcePoolInterface.
// A group's client should implement this interface.
type NodeConfigSourcePoolsGetter interface {
	NodeConfigSourcePools() NodeConfigSourcePoolInterface
}

// NodeConfigSourcePoolInterface has methods to work with NodeConfigSourcePool resources.
type NodeConfigSourcePoolInterface interface {
	Create(*v1alpha1.NodeConfigSourcePool) (*v1alpha1.NodeConfigSourcePool, error)
	Update(*v1alpha1.NodeConfigSourcePool) (*v1alpha1.NodeConfigSourcePool, error)
	Delete(name string, options *v1.DeleteOptions) error
	DeleteCollection(options *v1.DeleteOptions, listOptions v1.ListOptions) error
	Get(name string, options v1.GetOptions) (*v1alpha1.NodeConfigSourcePool, error)
	List(opts v1.ListOptions) (*v1alpha1.NodeConfigSourcePoolList, error)
	Watch(opts v1.ListOptions) (watch.Interface, error)
	Patch(name string, pt types.PatchType, data []byte, subresources ...string) (result *v1alpha1.NodeConfigSourcePool, err error)
	NodeConfigSourcePoolExpansion
}

// nodeConfigSourcePools implements NodeConfigSourcePoolInterface
type nodeConfigSourcePools struct {
	client rest.Interface
}

// newNodeConfigSourcePools returns a NodeConfigSourcePools
func newNodeConfigSourcePools(c *NodeconfigV1alpha1Client) *nodeConfigSourcePools {
	return &nodeConfigSourcePools{
		client: c.RESTClient(),
	}
}

// Get takes name of the nodeConfigSourcePool, and returns the corresponding nodeConfigSourcePool object, and an error if there is any.
func (c *nodeConfigSourcePools) Get(name string, options v1.GetOptions) (result *v1alpha1.NodeConfigSourcePool, err error) {
	result = &v1alpha1.NodeConfigSourcePool{}
	err = c.client.Get().
		Resource("nodeconfigsourcepools").
		Name(name).
		VersionedParams(&options, scheme.ParameterCodec).
		Do().
		Into(result)
	return
}

// List takes label and field selectors, and returns the list of NodeConfigSourcePools that match those selectors.
func (c *nodeConfigSourcePools) List(opts v1.ListOptions) (result *v1alpha1.NodeConfigSourcePoolList, err error) {
	result = &v1alpha1.NodeConfigSourcePoolList{}
	err = c.client.Get().
		Resource("nodeconfigsourcepools").
		VersionedParams(&opts, scheme.ParameterCodec).
		Do().
		Into(result)
	return
}

// Watch returns a watch.Interface that watches the requested nodeConfigSourcePools.
func (c *nodeConfigSourcePools) Watch(opts v1.ListOptions) (watch.Interface, error) {
	glog.Infof("watching nodeconfigsourcepools via generated client")
	opts.Watch = true
	return c.client.Get().
		Resource("nodeconfigsourcepools").
		VersionedParams(&opts, scheme.ParameterCodec).
		Watch()
}

// Create takes the representation of a nodeConfigSourcePool and creates it.  Returns the server's representation of the nodeConfigSourcePool, and an error, if there is any.
func (c *nodeConfigSourcePools) Create(nodeConfigSourcePool *v1alpha1.NodeConfigSourcePool) (result *v1alpha1.NodeConfigSourcePool, err error) {
	result = &v1alpha1.NodeConfigSourcePool{}
	err = c.client.Post().
		Resource("nodeconfigsourcepools").
		Body(nodeConfigSourcePool).
		Do().
		Into(result)
	return
}

// Update takes the representation of a nodeConfigSourcePool and updates it. Returns the server's representation of the nodeConfigSourcePool, and an error, if there is any.
func (c *nodeConfigSourcePools) Update(nodeConfigSourcePool *v1alpha1.NodeConfigSourcePool) (result *v1alpha1.NodeConfigSourcePool, err error) {
	result = &v1alpha1.NodeConfigSourcePool{}
	err = c.client.Put().
		Resource("nodeconfigsourcepools").
		Name(nodeConfigSourcePool.Name).
		Body(nodeConfigSourcePool).
		Do().
		Into(result)
	return
}

// Delete takes name of the nodeConfigSourcePool and deletes it. Returns an error if one occurs.
func (c *nodeConfigSourcePools) Delete(name string, options *v1.DeleteOptions) error {
	return c.client.Delete().
		Resource("nodeconfigsourcepools").
		Name(name).
		Body(options).
		Do().
		Error()
}

// DeleteCollection deletes a collection of objects.
func (c *nodeConfigSourcePools) DeleteCollection(options *v1.DeleteOptions, listOptions v1.ListOptions) error {
	return c.client.Delete().
		Resource("nodeconfigsourcepools").
		VersionedParams(&listOptions, scheme.ParameterCodec).
		Body(options).
		Do().
		Error()
}

// Patch applies the patch and returns the patched nodeConfigSourcePool.
func (c *nodeConfigSourcePools) Patch(name string, pt types.PatchType, data []byte, subresources ...string) (result *v1alpha1.NodeConfigSourcePool, err error) {
	result = &v1alpha1.NodeConfigSourcePool{}
	err = c.client.Patch(pt).
		Resource("nodeconfigsourcepools").
		SubResource(subresources...).
		Name(name).
		Body(data).
		Do().
		Into(result)
	return
}
