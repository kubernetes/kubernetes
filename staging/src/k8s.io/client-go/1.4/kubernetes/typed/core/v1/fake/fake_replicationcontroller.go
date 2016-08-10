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

package fake

import (
	api "k8s.io/client-go/1.4/pkg/api"
	unversioned "k8s.io/client-go/1.4/pkg/api/unversioned"
	v1 "k8s.io/client-go/1.4/pkg/api/v1"
	labels "k8s.io/client-go/1.4/pkg/labels"
	watch "k8s.io/client-go/1.4/pkg/watch"
	testing "k8s.io/client-go/1.4/testing"
)

// FakeReplicationControllers implements ReplicationControllerInterface
type FakeReplicationControllers struct {
	Fake *FakeCore
	ns   string
}

var replicationcontrollersResource = unversioned.GroupVersionResource{Group: "", Version: "v1", Resource: "replicationcontrollers"}

func (c *FakeReplicationControllers) Create(replicationController *v1.ReplicationController) (result *v1.ReplicationController, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewCreateAction(replicationcontrollersResource, c.ns, replicationController), &v1.ReplicationController{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1.ReplicationController), err
}

func (c *FakeReplicationControllers) Update(replicationController *v1.ReplicationController) (result *v1.ReplicationController, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewUpdateAction(replicationcontrollersResource, c.ns, replicationController), &v1.ReplicationController{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1.ReplicationController), err
}

func (c *FakeReplicationControllers) UpdateStatus(replicationController *v1.ReplicationController) (*v1.ReplicationController, error) {
	obj, err := c.Fake.
		Invokes(testing.NewUpdateSubresourceAction(replicationcontrollersResource, "status", c.ns, replicationController), &v1.ReplicationController{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1.ReplicationController), err
}

func (c *FakeReplicationControllers) Delete(name string, options *api.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(testing.NewDeleteAction(replicationcontrollersResource, c.ns, name), &v1.ReplicationController{})

	return err
}

func (c *FakeReplicationControllers) DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error {
	action := testing.NewDeleteCollectionAction(replicationcontrollersResource, c.ns, listOptions)

	_, err := c.Fake.Invokes(action, &v1.ReplicationControllerList{})
	return err
}

func (c *FakeReplicationControllers) Get(name string) (result *v1.ReplicationController, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewGetAction(replicationcontrollersResource, c.ns, name), &v1.ReplicationController{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1.ReplicationController), err
}

func (c *FakeReplicationControllers) List(opts api.ListOptions) (result *v1.ReplicationControllerList, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewListAction(replicationcontrollersResource, c.ns, opts), &v1.ReplicationControllerList{})

	if obj == nil {
		return nil, err
	}

	label := opts.LabelSelector
	if label == nil {
		label = labels.Everything()
	}
	list := &v1.ReplicationControllerList{}
	for _, item := range obj.(*v1.ReplicationControllerList).Items {
		if label.Matches(labels.Set(item.Labels)) {
			list.Items = append(list.Items, item)
		}
	}
	return list, err
}

// Watch returns a watch.Interface that watches the requested replicationControllers.
func (c *FakeReplicationControllers) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(testing.NewWatchAction(replicationcontrollersResource, c.ns, opts))

}

// Patch applies the patch and returns the patched replicationController.
func (c *FakeReplicationControllers) Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *v1.ReplicationController, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewPatchSubresourceAction(replicationcontrollersResource, c.ns, name, data, subresources...), &v1.ReplicationController{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1.ReplicationController), err
}
