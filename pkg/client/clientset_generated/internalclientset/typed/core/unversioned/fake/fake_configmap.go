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
	api "k8s.io/kubernetes/pkg/api"
	unversioned "k8s.io/kubernetes/pkg/api/unversioned"
	core "k8s.io/kubernetes/pkg/client/testing/core"
	labels "k8s.io/kubernetes/pkg/labels"
	watch "k8s.io/kubernetes/pkg/watch"
)

// FakeConfigMaps implements ConfigMapInterface
type FakeConfigMaps struct {
	Fake *FakeCore
	ns   string
}

var configmapsResource = unversioned.GroupVersionResource{Group: "", Version: "", Resource: "configmaps"}

func (c *FakeConfigMaps) Create(configMap *api.ConfigMap) (result *api.ConfigMap, err error) {
	obj, err := c.Fake.
		Invokes(core.NewCreateAction(configmapsResource, c.ns, configMap), &api.ConfigMap{})

	if obj == nil {
		return nil, err
	}
	return obj.(*api.ConfigMap), err
}

func (c *FakeConfigMaps) Update(configMap *api.ConfigMap) (result *api.ConfigMap, err error) {
	obj, err := c.Fake.
		Invokes(core.NewUpdateAction(configmapsResource, c.ns, configMap), &api.ConfigMap{})

	if obj == nil {
		return nil, err
	}
	return obj.(*api.ConfigMap), err
}

func (c *FakeConfigMaps) Delete(name string, options *api.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(core.NewDeleteAction(configmapsResource, c.ns, name), &api.ConfigMap{})

	return err
}

func (c *FakeConfigMaps) DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error {
	action := core.NewDeleteCollectionAction(configmapsResource, c.ns, listOptions)

	_, err := c.Fake.Invokes(action, &api.ConfigMapList{})
	return err
}

func (c *FakeConfigMaps) Get(name string) (result *api.ConfigMap, err error) {
	obj, err := c.Fake.
		Invokes(core.NewGetAction(configmapsResource, c.ns, name), &api.ConfigMap{})

	if obj == nil {
		return nil, err
	}
	return obj.(*api.ConfigMap), err
}

func (c *FakeConfigMaps) List(opts api.ListOptions) (result *api.ConfigMapList, err error) {
	obj, err := c.Fake.
		Invokes(core.NewListAction(configmapsResource, c.ns, opts), &api.ConfigMapList{})

	if obj == nil {
		return nil, err
	}

	label := opts.LabelSelector
	if label == nil {
		label = labels.Everything()
	}
	list := &api.ConfigMapList{}
	for _, item := range obj.(*api.ConfigMapList).Items {
		if label.Matches(labels.Set(item.Labels)) {
			list.Items = append(list.Items, item)
		}
	}
	return list, err
}

// Watch returns a watch.Interface that watches the requested configMaps.
func (c *FakeConfigMaps) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(core.NewWatchAction(configmapsResource, c.ns, opts))

}

// Patch applies the patch and returns the patched configMap.
func (c *FakeConfigMaps) Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *api.ConfigMap, err error) {
	obj, err := c.Fake.
		Invokes(core.NewPatchSubresourceAction(configmapsResource, c.ns, name, data, subresources...), &api.ConfigMap{})

	if obj == nil {
		return nil, err
	}
	return obj.(*api.ConfigMap), err
}
