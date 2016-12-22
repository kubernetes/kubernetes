/*
Copyright 2015 The Kubernetes Authors.

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
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/client/testing/core"
	"k8s.io/kubernetes/pkg/runtime/schema"
)

func (c *FakeScales) Get(kind string, name string) (result *extensions.Scale, err error) {
	action := core.GetActionImpl{}
	action.Verb = "get"
	action.Namespace = c.ns
	action.Resource = schema.GroupVersionResource{Resource: kind}
	action.Subresource = "scale"
	action.Name = name
	obj, err := c.Fake.Invokes(action, &extensions.Scale{})
	result = obj.(*extensions.Scale)
	return
}

func (c *FakeScales) Update(kind string, scale *extensions.Scale) (result *extensions.Scale, err error) {
	action := core.UpdateActionImpl{}
	action.Verb = "update"
	action.Namespace = c.ns
	action.Resource = schema.GroupVersionResource{Resource: kind}
	action.Subresource = "scale"
	action.Object = scale
	obj, err := c.Fake.Invokes(action, scale)
	result = obj.(*extensions.Scale)
	return
}
