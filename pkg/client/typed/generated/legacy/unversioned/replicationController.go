/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	api "k8s.io/kubernetes/pkg/api"
	unversioned "k8s.io/kubernetes/pkg/api/unversioned"
	watch "k8s.io/kubernetes/pkg/watch"
)

// ReplicationControllerNamespacer has methods to work with ReplicationController resources in a namespace
type ReplicationControllerNamespacer interface {
	ReplicationControllers(namespace string) ReplicationControllerInterface
}

// ReplicationControllerInterface has methods to work with ReplicationController resources.
type ReplicationControllerInterface interface {
	Create(*api.ReplicationController) (*api.ReplicationController, error)
	Update(*api.ReplicationController) (*api.ReplicationController, error)
	Delete(name string, options *api.DeleteOptions) error
	Get(name string) (*api.ReplicationController, error)
	List(opts unversioned.ListOptions) (*api.ReplicationControllerList, error)
	Watch(opts unversioned.ListOptions) (watch.Interface, error)
}

// replicationControllers implements ReplicationControllerInterface
type replicationControllers struct {
	client *LegacyClient
	ns     string
}

// newReplicationControllers returns a ReplicationControllers
func newReplicationControllers(c *LegacyClient, namespace string) *replicationControllers {
	return &replicationControllers{
		client: c,
		ns:     namespace,
	}
}

// Create takes the representation of a replicationController and creates it.  Returns the server's representation of the replicationController, and an error, if there is any.
func (c *replicationControllers) Create(replicationController *api.ReplicationController) (result *api.ReplicationController, err error) {
	result = &api.ReplicationController{}
	err = c.client.Post().
		Namespace(c.ns).
		Resource("replicationControllers").
		Body(replicationController).
		Do().
		Into(result)
	return
}

// Update takes the representation of a replicationController and updates it. Returns the server's representation of the replicationController, and an error, if there is any.
func (c *replicationControllers) Update(replicationController *api.ReplicationController) (result *api.ReplicationController, err error) {
	result = &api.ReplicationController{}
	err = c.client.Put().
		Namespace(c.ns).
		Resource("replicationControllers").
		Name(replicationController.Name).
		Body(replicationController).
		Do().
		Into(result)
	return
}

// Delete takes name of the replicationController and deletes it. Returns an error if one occurs.
func (c *replicationControllers) Delete(name string, options *api.DeleteOptions) error {
	if options == nil {
		return c.client.Delete().Namespace(c.ns).Resource("replicationControllers").Name(name).Do().Error()
	}
	body, err := api.Scheme.EncodeToVersion(options, c.client.APIVersion().String())
	if err != nil {
		return err
	}
	return c.client.Delete().
		Namespace(c.ns).
		Resource("replicationControllers").
		Name(name).
		Body(body).
		Do().
		Error()
}

// Get takes name of the replicationController, and returns the corresponding replicationController object, and an error if there is any.
func (c *replicationControllers) Get(name string) (result *api.ReplicationController, err error) {
	result = &api.ReplicationController{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("replicationControllers").
		Name(name).
		Do().
		Into(result)
	return
}

// List takes label and field selectors, and returns the list of ReplicationControllers that match those selectors.
func (c *replicationControllers) List(opts unversioned.ListOptions) (result *api.ReplicationControllerList, err error) {
	result = &api.ReplicationControllerList{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("replicationControllers").
		VersionedParams(&opts, api.Scheme).
		Do().
		Into(result)
	return
}

// Watch returns a watch.Interface that watches the requested replicationControllers.
func (c *replicationControllers) Watch(opts unversioned.ListOptions) (watch.Interface, error) {
	return c.client.Get().
		Prefix("watch").
		Namespace(c.ns).
		Resource("replicationControllers").
		VersionedParams(&opts, api.Scheme).
		Watch()
}
