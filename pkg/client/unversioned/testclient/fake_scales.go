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
	"k8s.io/kubernetes/pkg/apis/experimental"
)

// FakeScales implements ScaleInterface. Meant to be embedded into a struct to get a default
// implementation. This makes faking out just the methods you want to test easier.
type FakeScales struct {
	Fake      *FakeExperimental
	Namespace string
}

func (c *FakeScales) Get(kind string, name string) (result *experimental.Scale, err error) {
	action := GetActionImpl{}
	action.Verb = "get"
	action.Namespace = c.Namespace
	action.Resource = kind
	action.Subresource = "scale"
	action.Name = name
	obj, err := c.Fake.Invokes(action, &experimental.Scale{})
	result = obj.(*experimental.Scale)
	return
}

func (c *FakeScales) Update(kind string, scale *experimental.Scale) (result *experimental.Scale, err error) {
	action := UpdateActionImpl{}
	action.Verb = "update"
	action.Namespace = c.Namespace
	action.Resource = kind
	action.Subresource = "scale"
	action.Object = scale
	obj, err := c.Fake.Invokes(action, scale)
	result = obj.(*experimental.Scale)
	return
}
