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

package testclient

import (
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/watch"
)

// FakePodTemplates implements PodTemplatesInterface. Meant to be embedded into a struct to get a default
// implementation. This makes faking out just the methods you want to test easier.
type FakePodTemplates struct {
	Fake      *Fake
	Namespace string
}

func (c *FakePodTemplates) Get(name string) (*api.PodTemplate, error) {
	obj, err := c.Fake.Invokes(NewGetAction("podtemplates", c.Namespace, name), &api.PodTemplate{})
	if obj == nil {
		return nil, err
	}

	return obj.(*api.PodTemplate), err
}

func (c *FakePodTemplates) List(opts api.ListOptions) (*api.PodTemplateList, error) {
	obj, err := c.Fake.Invokes(NewListAction("podtemplates", c.Namespace, opts), &api.PodTemplateList{})
	if obj == nil {
		return nil, err
	}

	return obj.(*api.PodTemplateList), err
}

func (c *FakePodTemplates) Create(pod *api.PodTemplate) (*api.PodTemplate, error) {
	obj, err := c.Fake.Invokes(NewCreateAction("podtemplates", c.Namespace, pod), pod)
	if obj == nil {
		return nil, err
	}

	return obj.(*api.PodTemplate), err
}

func (c *FakePodTemplates) Update(pod *api.PodTemplate) (*api.PodTemplate, error) {
	obj, err := c.Fake.Invokes(NewUpdateAction("podtemplates", c.Namespace, pod), pod)
	if obj == nil {
		return nil, err
	}

	return obj.(*api.PodTemplate), err
}

func (c *FakePodTemplates) Delete(name string, options *api.DeleteOptions) error {
	_, err := c.Fake.Invokes(NewDeleteAction("podtemplates", c.Namespace, name), &api.PodTemplate{})
	return err
}

func (c *FakePodTemplates) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.Fake.InvokesWatch(NewWatchAction("podtemplates", c.Namespace, opts))
}
