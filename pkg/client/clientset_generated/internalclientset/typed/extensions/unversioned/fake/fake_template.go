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

package fake

import (
	api "k8s.io/kubernetes/pkg/api"
	unversioned "k8s.io/kubernetes/pkg/api/unversioned"
	extensions "k8s.io/kubernetes/pkg/apis/extensions"
	core "k8s.io/kubernetes/pkg/client/testing/core"
	labels "k8s.io/kubernetes/pkg/labels"
	watch "k8s.io/kubernetes/pkg/watch"
)

// FakeTemplates implements TemplateInterface
type FakeTemplates struct {
	Fake *FakeExtensions
	ns   string
}

var templatesResource = unversioned.GroupVersionResource{Group: "extensions", Version: "", Resource: "templates"}

func (c *FakeTemplates) Create(template *extensions.Template) (result *extensions.Template, err error) {
	obj, err := c.Fake.
		Invokes(core.NewCreateAction(templatesResource, c.ns, template), &extensions.Template{})

	if obj == nil {
		return nil, err
	}
	return obj.(*extensions.Template), err
}

func (c *FakeTemplates) Update(template *extensions.Template) (result *extensions.Template, err error) {
	obj, err := c.Fake.
		Invokes(core.NewUpdateAction(templatesResource, c.ns, template), &extensions.Template{})

	if obj == nil {
		return nil, err
	}
	return obj.(*extensions.Template), err
}

func (c *FakeTemplates) Delete(name string, options *api.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(core.NewDeleteAction(templatesResource, c.ns, name), &extensions.Template{})

	return err
}

func (c *FakeTemplates) DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error {
	action := core.NewDeleteCollectionAction(templatesResource, c.ns, listOptions)

	_, err := c.Fake.Invokes(action, &extensions.TemplateList{})
	return err
}

func (c *FakeTemplates) Get(name string) (result *extensions.Template, err error) {
	obj, err := c.Fake.
		Invokes(core.NewGetAction(templatesResource, c.ns, name), &extensions.Template{})

	if obj == nil {
		return nil, err
	}
	return obj.(*extensions.Template), err
}

func (c *FakeTemplates) List(opts api.ListOptions) (result *extensions.TemplateList, err error) {
	obj, err := c.Fake.
		Invokes(core.NewListAction(templatesResource, c.ns, opts), &extensions.TemplateList{})

	if obj == nil {
		return nil, err
	}

	label := opts.LabelSelector
	if label == nil {
		label = labels.Everything()
	}
	list := &extensions.TemplateList{}
	for _, item := range obj.(*extensions.TemplateList).Items {
		if label.Matches(labels.Set(item.Labels)) {
			list.Items = append(list.Items, item)
		}
	}
	return list, err
}

// Watch returns a watch.Interface that watches the requested templates.
func (c *FakeTemplates) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(core.NewWatchAction(templatesResource, c.ns, opts))

}
