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

package unversioned

import (
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/watch"
)

// ReplicaSetsNamespacer has methods to work with ReplicaSet resources in a namespace
type ReplicaSetsNamespacer interface {
	ReplicaSets(namespace string) ReplicaSetInterface
}

// ReplicaSetInterface has methods to work with ReplicaSet resources.
type ReplicaSetInterface interface {
	List(opts api.ListOptions) (*extensions.ReplicaSetList, error)
	Get(name string) (*extensions.ReplicaSet, error)
	Create(ctrl *extensions.ReplicaSet) (*extensions.ReplicaSet, error)
	Update(ctrl *extensions.ReplicaSet) (*extensions.ReplicaSet, error)
	UpdateStatus(ctrl *extensions.ReplicaSet) (*extensions.ReplicaSet, error)
	Delete(name string, options *api.DeleteOptions) error
	Watch(opts api.ListOptions) (watch.Interface, error)
}

// replicaSets implements ReplicaSetsNamespacer interface
type replicaSets struct {
	client *ExtensionsClient
	ns     string
}

// newReplicaSets returns a ReplicaSetClient
func newReplicaSets(c *ExtensionsClient, namespace string) *replicaSets {
	return &replicaSets{c, namespace}
}

// List takes a selector, and returns the list of ReplicaSets that match that selector.
func (c *replicaSets) List(opts api.ListOptions) (result *extensions.ReplicaSetList, err error) {
	result = &extensions.ReplicaSetList{}
	err = c.client.Get().Namespace(c.ns).Resource("replicasets").VersionedParams(&opts, api.ParameterCodec).Do().Into(result)
	return
}

// Get returns information about a particular ReplicaSet.
func (c *replicaSets) Get(name string) (result *extensions.ReplicaSet, err error) {
	result = &extensions.ReplicaSet{}
	err = c.client.Get().Namespace(c.ns).Resource("replicasets").Name(name).Do().Into(result)
	return
}

// Create creates a new ReplicaSet.
func (c *replicaSets) Create(rs *extensions.ReplicaSet) (result *extensions.ReplicaSet, err error) {
	result = &extensions.ReplicaSet{}
	err = c.client.Post().Namespace(c.ns).Resource("replicasets").Body(rs).Do().Into(result)
	return
}

// Update updates an existing ReplicaSet.
func (c *replicaSets) Update(rs *extensions.ReplicaSet) (result *extensions.ReplicaSet, err error) {
	result = &extensions.ReplicaSet{}
	err = c.client.Put().Namespace(c.ns).Resource("replicasets").Name(rs.Name).Body(rs).Do().Into(result)
	return
}

// UpdateStatus updates an existing ReplicaSet status
func (c *replicaSets) UpdateStatus(rs *extensions.ReplicaSet) (result *extensions.ReplicaSet, err error) {
	result = &extensions.ReplicaSet{}
	err = c.client.Put().Namespace(c.ns).Resource("replicasets").Name(rs.Name).SubResource("status").Body(rs).Do().Into(result)
	return
}

// Delete deletes an existing ReplicaSet.
func (c *replicaSets) Delete(name string, options *api.DeleteOptions) (err error) {
	return c.client.Delete().Namespace(c.ns).Resource("replicasets").Name(name).Body(options).Do().Error()
}

// Watch returns a watch.Interface that watches the requested ReplicaSets.
func (c *replicaSets) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.client.Get().
		Prefix("watch").
		Namespace(c.ns).
		Resource("replicasets").
		VersionedParams(&opts, api.ParameterCodec).
		Watch()
}
