/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/controlplane"
	"k8s.io/kubernetes/pkg/watch"
)

type ClustersInterface interface {
	Clusters() ClusterInterface
}

type ClusterInterface interface {
	Get(name string) (result *controlplane.Cluster, err error)
	Create(cluster *controlplane.Cluster) (*controlplane.Cluster, error)
	List(opts api.ListOptions) (*controlplane.ClusterList, error)
	Delete(name string) error
	Update(*controlplane.Cluster) (*controlplane.Cluster, error)
	UpdateStatus(*controlplane.Cluster) (*controlplane.Cluster, error)
	Watch(opts api.ListOptions) (watch.Interface, error)
}

// clusters implements clustersInterface
type clusters struct {
	r *Client
}

// newClusters returns a clusters object.
func newClusters(c *Client) *clusters {
	return &clusters{c}
}

// resourceName returns cluster's URL resource name.
func (c *clusters) resourceName() string {
	return "clusters"
}

// Create creates a new cluster.
func (c *clusters) Create(cluster *controlplane.Cluster) (*controlplane.Cluster, error) {
	result := &controlplane.Cluster{}
	err := c.r.Post().Resource(c.resourceName()).Body(cluster).Do().Into(result)
	return result, err
}

// List takes a selector, and returns the list of clusters that match that selector in the cluster.
func (c *clusters) List(opts api.ListOptions) (*controlplane.ClusterList, error) {
	result := &controlplane.ClusterList{}
	err := c.r.Get().Resource(c.resourceName()).VersionedParams(&opts, api.ParameterCodec).Do().Into(result)
	return result, err
}

// Get gets an existing cluster.
func (c *clusters) Get(name string) (*controlplane.Cluster, error) {
	result := &controlplane.Cluster{}
	err := c.r.Get().Resource(c.resourceName()).Name(name).Do().Into(result)
	return result, err
}

// Delete deletes an existing cluster.
func (c *clusters) Delete(name string) error {
	return c.r.Delete().Resource(c.resourceName()).Name(name).Do().Error()
}

// Update updates an existing cluster.
func (c *clusters) Update(cluster *controlplane.Cluster) (*controlplane.Cluster, error) {
	result := &controlplane.Cluster{}
	err := c.r.Put().Resource(c.resourceName()).Name(cluster.Name).Body(cluster).Do().Into(result)
	return result, err
}

func (c *clusters) UpdateStatus(cluster *controlplane.Cluster) (*controlplane.Cluster, error) {
	result := &controlplane.Cluster{}
	err := c.r.Put().Resource(c.resourceName()).Name(cluster.Name).SubResource("status").Body(cluster).Do().Into(result)
	return result, err
}

// Watch returns a watch.Interface that watches the requested clusters.
func (c *clusters) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.r.Get().
		Prefix("watch").
		Namespace(api.NamespaceAll).
		Resource(c.resourceName()).
		VersionedParams(&opts, api.ParameterCodec).
		Watch()
}
