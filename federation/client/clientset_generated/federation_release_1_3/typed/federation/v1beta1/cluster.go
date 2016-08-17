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
	v1beta1 "k8s.io/kubernetes/federation/apis/federation/v1beta1"
	api "k8s.io/kubernetes/pkg/api"
	watch "k8s.io/kubernetes/pkg/watch"
)

// ClustersGetter has a method to return a ClusterInterface.
// A group's client should implement this interface.
type ClustersGetter interface {
	Clusters() ClusterInterface
}

// ClusterInterface has methods to work with Cluster resources.
type ClusterInterface interface {
	Create(*v1beta1.Cluster) (*v1beta1.Cluster, error)
	Update(*v1beta1.Cluster) (*v1beta1.Cluster, error)
	UpdateStatus(*v1beta1.Cluster) (*v1beta1.Cluster, error)
	Delete(name string, options *api.DeleteOptions) error
	DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error
	Get(name string) (*v1beta1.Cluster, error)
	List(opts api.ListOptions) (*v1beta1.ClusterList, error)
	Watch(opts api.ListOptions) (watch.Interface, error)
	Patch(name string, pt api.PatchType, data []byte) (result *v1beta1.Cluster, err error)
	ClusterExpansion
}

// clusters implements ClusterInterface
type clusters struct {
	client *FederationClient
}

// newClusters returns a Clusters
func newClusters(c *FederationClient) *clusters {
	return &clusters{
		client: c,
	}
}

// Create takes the representation of a cluster and creates it.  Returns the server's representation of the cluster, and an error, if there is any.
func (c *clusters) Create(cluster *v1beta1.Cluster) (result *v1beta1.Cluster, err error) {
	result = &v1beta1.Cluster{}
	err = c.client.Post().
		Resource("clusters").
		Body(cluster).
		Do().
		Into(result)
	return
}

// Update takes the representation of a cluster and updates it. Returns the server's representation of the cluster, and an error, if there is any.
func (c *clusters) Update(cluster *v1beta1.Cluster) (result *v1beta1.Cluster, err error) {
	result = &v1beta1.Cluster{}
	err = c.client.Put().
		Resource("clusters").
		Name(cluster.Name).
		Body(cluster).
		Do().
		Into(result)
	return
}

func (c *clusters) UpdateStatus(cluster *v1beta1.Cluster) (result *v1beta1.Cluster, err error) {
	result = &v1beta1.Cluster{}
	err = c.client.Put().
		Resource("clusters").
		Name(cluster.Name).
		SubResource("status").
		Body(cluster).
		Do().
		Into(result)
	return
}

// Delete takes name of the cluster and deletes it. Returns an error if one occurs.
func (c *clusters) Delete(name string, options *api.DeleteOptions) error {
	return c.client.Delete().
		Resource("clusters").
		Name(name).
		Body(options).
		Do().
		Error()
}

// DeleteCollection deletes a collection of objects.
func (c *clusters) DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error {
	return c.client.Delete().
		Resource("clusters").
		VersionedParams(&listOptions, api.ParameterCodec).
		Body(options).
		Do().
		Error()
}

// Get takes name of the cluster, and returns the corresponding cluster object, and an error if there is any.
func (c *clusters) Get(name string) (result *v1beta1.Cluster, err error) {
	result = &v1beta1.Cluster{}
	err = c.client.Get().
		Resource("clusters").
		Name(name).
		Do().
		Into(result)
	return
}

// List takes label and field selectors, and returns the list of Clusters that match those selectors.
func (c *clusters) List(opts api.ListOptions) (result *v1beta1.ClusterList, err error) {
	result = &v1beta1.ClusterList{}
	err = c.client.Get().
		Resource("clusters").
		VersionedParams(&opts, api.ParameterCodec).
		Do().
		Into(result)
	return
}

// Watch returns a watch.Interface that watches the requested clusters.
func (c *clusters) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.client.Get().
		Prefix("watch").
		Resource("clusters").
		VersionedParams(&opts, api.ParameterCodec).
		Watch()
}

// Patch applies the patch and returns the patched cluster.
func (c *clusters) Patch(name string, pt api.PatchType, data []byte) (result *v1beta1.Cluster, err error) {
	result = &v1beta1.Cluster{}
	err = c.client.Patch(pt).
		Resource("clusters").
		Name(name).
		Body(data).
		Do().
		Into(result)
	return
}
