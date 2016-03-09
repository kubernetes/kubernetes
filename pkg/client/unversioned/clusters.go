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

package unversioned

import (
	"k8s.io/kubernetes/pkg/api"
	apiclusters "k8s.io/kubernetes/pkg/apis/clusters"
	"k8s.io/kubernetes/pkg/watch"
)

type ClustersInterface interface {
	Clusters() ClusterInterface
}

type ClusterInterface interface {
	Get(name string) (*apiclusters.Cluster, error)
	Create(cluster *apiclusters.Cluster) (*apiclusters.Cluster, error)
	List(opts api.ListOptions) (*apiclusters.ClusterList, error)
	Delete(name string) error
	Update(*apiclusters.Cluster) (*apiclusters.Cluster, error)
	UpdateStatus(*apiclusters.Cluster) (*apiclusters.Cluster, error)
	Watch(opts api.ListOptions) (watch.Interface, error)
}

type clusters struct {
	r *Client
}

func newClusters(c *Client) *clusters {
	return &clusters{c}
}

func (c *clusters) resourceName() string {
	return "clusters"
}

func (c *clusters) Get(name string) (*apiclusters.Cluster, error) {
	result := &apiclusters.Cluster{}
	err := c.r.Get().Resource(c.resourceName()).Name(name).Do().Into(result)
	return result, err
}

func (c *clusters) Create(cluster *apiclusters.Cluster) (*apiclusters.Cluster, error) {
	result := &apiclusters.Cluster{}
	err := c.r.Post().Resource(c.resourceName()).Body(cluster).Do().Into(result)
	return result, err
}

func (c *clusters) List(opts api.ListOptions) (*apiclusters.ClusterList, error) {
	result := &apiclusters.ClusterList{}
	err := c.r.Get().Resource(c.resourceName()).VersionedParams(&opts, api.ParameterCodec).Do().Into(result)
	return result, err
}

func (c *clusters) Delete(name string) error {
	return c.r.Delete().Resource(c.resourceName()).Name(name).Do().Error()
}

func (c *clusters) Update(cluster *apiclusters.Cluster) (*apiclusters.Cluster, error) {
	result := &apiclusters.Cluster{}
	err := c.r.Put().Resource(c.resourceName()).Name(cluster.Name).Body(cluster).Do().Into(result)
	return result, err
}

func (c *clusters) UpdateStatus(cluster *apiclusters.Cluster) (*apiclusters.Cluster, error) {
	result := &apiclusters.Cluster{}
	err := c.r.Put().Resource(c.resourceName()).Name(cluster.Name).SubResource("status").Body(cluster).Do().Into(result)
	return result, err
}

// Watch returns a watch.Interface that watches the requested cluster.
func (c *clusters) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.r.Get().
		Prefix("watch").
		Namespace(api.NamespaceAll).
		Resource(c.resourceName()).
		VersionedParams(&opts, api.ParameterCodec).
		Watch()
}
