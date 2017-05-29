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

package fake

import (
	meta_v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	labels "k8s.io/apimachinery/pkg/labels"
	schema "k8s.io/apimachinery/pkg/runtime/schema"
	types "k8s.io/apimachinery/pkg/types"
	watch "k8s.io/apimachinery/pkg/watch"
	testing "k8s.io/client-go/testing"
	v1 "k8s.io/kubernetes/pkg/api/v1"
)

// FakeConfigMaps implements ConfigMapInterface
type FakeConfigMaps struct {
	Fake *FakeCoreV1
	ns   string
}

var configmapsResource = schema.GroupVersionResource{Group: "", Version: "v1", Resource: "configmaps"}

var configmapsKind = schema.GroupVersionKind{Group: "", Version: "v1", Kind: "ConfigMap"}

func (c *FakeConfigMaps) Create(configMap *v1.ConfigMap) (result *v1.ConfigMap, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewCreateAction(configmapsResource, c.ns, configMap), &v1.ConfigMap{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1.ConfigMap), err
}

func (c *FakeConfigMaps) Update(configMap *v1.ConfigMap) (result *v1.ConfigMap, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewUpdateAction(configmapsResource, c.ns, configMap), &v1.ConfigMap{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1.ConfigMap), err
}

func (c *FakeConfigMaps) Delete(name string, options *meta_v1.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(testing.NewDeleteAction(configmapsResource, c.ns, name), &v1.ConfigMap{})

	return err
}

func (c *FakeConfigMaps) DeleteCollection(options *meta_v1.DeleteOptions, listOptions meta_v1.ListOptions) error {
	action := testing.NewDeleteCollectionAction(configmapsResource, c.ns, listOptions)

	_, err := c.Fake.Invokes(action, &v1.ConfigMapList{})
	return err
}

func (c *FakeConfigMaps) Get(name string, options meta_v1.GetOptions) (result *v1.ConfigMap, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewGetAction(configmapsResource, c.ns, name), &v1.ConfigMap{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1.ConfigMap), err
}

func (c *FakeConfigMaps) List(opts meta_v1.ListOptions) (result *v1.ConfigMapList, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewListAction(configmapsResource, configmapsKind, c.ns, opts), &v1.ConfigMapList{})

	if obj == nil {
		return nil, err
	}

	label, _, _ := testing.ExtractFromListOptions(opts)
	if label == nil {
		label = labels.Everything()
	}
	list := &v1.ConfigMapList{}
	for _, item := range obj.(*v1.ConfigMapList).Items {
		if label.Matches(labels.Set(item.Labels)) {
			list.Items = append(list.Items, item)
		}
	}
	return list, err
}

// Watch returns a watch.Interface that watches the requested configMaps.
func (c *FakeConfigMaps) Watch(opts meta_v1.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(testing.NewWatchAction(configmapsResource, c.ns, opts))

}

// Patch applies the patch and returns the patched configMap.
func (c *FakeConfigMaps) Patch(name string, pt types.PatchType, data []byte, subresources ...string) (result *v1.ConfigMap, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewPatchSubresourceAction(configmapsResource, c.ns, name, data, subresources...), &v1.ConfigMap{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1.ConfigMap), err
}
