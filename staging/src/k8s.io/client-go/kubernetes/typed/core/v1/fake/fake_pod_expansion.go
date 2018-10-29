/*
Copyright 2014 The Kubernetes Authors.

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
	"k8s.io/api/core/v1"
	policy "k8s.io/api/policy/v1beta1"
	restclient "k8s.io/client-go/rest"
	core "k8s.io/client-go/testing"
)

func (c *FakePods) Bind(binding *v1.Binding) error {
	action := core.CreateActionImpl{}
	action.Verb = "create"
	action.Namespace = binding.Namespace
	action.Resource = podsResource
	action.Subresource = "binding"
	action.Object = binding

	_, err := c.Fake.Invokes(action, binding)
	return err
}

func (c *FakePods) GetBinding(name string) (result *v1.Binding, err error) {
	obj, err := c.Fake.
		Invokes(core.NewGetSubresourceAction(podsResource, c.ns, "binding", name), &v1.Binding{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1.Binding), err
}

func (c *FakePods) GetLogs(name string, opts *v1.PodLogOptions) *restclient.Request {
	action := core.GenericActionImpl{}
	action.Verb = "get"
	action.Namespace = c.ns
	action.Resource = podsResource
	action.Subresource = "log"
	action.Value = opts

	_, _ = c.Fake.Invokes(action, &v1.Pod{})
	return &restclient.Request{}
}

func (c *FakePods) Evict(eviction *policy.Eviction) error {
	action := core.CreateActionImpl{}
	action.Verb = "create"
	action.Resource = podsResource
	action.Subresource = "eviction"
	action.Object = eviction

	_, err := c.Fake.Invokes(action, eviction)
	return err
}
