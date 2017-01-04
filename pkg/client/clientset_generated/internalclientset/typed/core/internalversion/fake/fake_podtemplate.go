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
	api "k8s.io/kubernetes/pkg/api"
	unversioned "k8s.io/kubernetes/pkg/api/unversioned"
	core "k8s.io/kubernetes/pkg/client/testing/core"
	labels "k8s.io/kubernetes/pkg/labels"
	watch "k8s.io/kubernetes/pkg/watch"
)

// FakePodTemplates implements PodTemplateInterface
type FakePodTemplates struct {
	Fake *FakeCore
	ns   string
}

var podtemplatesResource = unversioned.GroupVersionResource{Group: "", Version: "", Resource: "podtemplates"}

func (c *FakePodTemplates) Create(podTemplate *api.PodTemplate) (result *api.PodTemplate, err error) {
	obj, err := c.Fake.
		Invokes(core.NewCreateAction(podtemplatesResource, c.ns, podTemplate), &api.PodTemplate{})

	if obj == nil {
		return nil, err
	}
	return obj.(*api.PodTemplate), err
}

func (c *FakePodTemplates) Update(podTemplate *api.PodTemplate) (result *api.PodTemplate, err error) {
	obj, err := c.Fake.
		Invokes(core.NewUpdateAction(podtemplatesResource, c.ns, podTemplate), &api.PodTemplate{})

	if obj == nil {
		return nil, err
	}
	return obj.(*api.PodTemplate), err
}

func (c *FakePodTemplates) Delete(name string, options *api.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(core.NewDeleteAction(podtemplatesResource, c.ns, name), &api.PodTemplate{})

	return err
}

func (c *FakePodTemplates) DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error {
	action := core.NewDeleteCollectionAction(podtemplatesResource, c.ns, listOptions)

	_, err := c.Fake.Invokes(action, &api.PodTemplateList{})
	return err
}

func (c *FakePodTemplates) Get(name string) (result *api.PodTemplate, err error) {
	obj, err := c.Fake.
		Invokes(core.NewGetAction(podtemplatesResource, c.ns, name), &api.PodTemplate{})

	if obj == nil {
		return nil, err
	}
	return obj.(*api.PodTemplate), err
}

func (c *FakePodTemplates) List(opts api.ListOptions) (result *api.PodTemplateList, err error) {
	obj, err := c.Fake.
		Invokes(core.NewListAction(podtemplatesResource, c.ns, opts), &api.PodTemplateList{})

	if obj == nil {
		return nil, err
	}

	label, _, _ := core.ExtractFromListOptions(opts)
	if label == nil {
		label = labels.Everything()
	}
	list := &api.PodTemplateList{}
	for _, item := range obj.(*api.PodTemplateList).Items {
		if label.Matches(labels.Set(item.Labels)) {
			list.Items = append(list.Items, item)
		}
	}
	return list, err
}

// Watch returns a watch.Interface that watches the requested podTemplates.
func (c *FakePodTemplates) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(core.NewWatchAction(podtemplatesResource, c.ns, opts))

}

// Patch applies the patch and returns the patched podTemplate.
func (c *FakePodTemplates) Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *api.PodTemplate, err error) {
	obj, err := c.Fake.
		Invokes(core.NewPatchSubresourceAction(podtemplatesResource, c.ns, name, data, subresources...), &api.PodTemplate{})

	if obj == nil {
		return nil, err
	}
	return obj.(*api.PodTemplate), err
}
