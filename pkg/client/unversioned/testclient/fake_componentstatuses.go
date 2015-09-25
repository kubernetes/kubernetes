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
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
)

// Fake implements ComponentStatusInterface.
type FakeComponentStatuses struct {
	Fake *Fake
}

func (c *FakeComponentStatuses) Get(name string) (*api.ComponentStatus, error) {
	obj, err := c.Fake.Invokes(NewRootGetAction("componentstatuses", name), &api.ComponentStatus{})
	if obj == nil {
		return nil, err
	}

	return obj.(*api.ComponentStatus), err
}

func (c *FakeComponentStatuses) List(label labels.Selector, field fields.Selector) (result *api.ComponentStatusList, err error) {
	obj, err := c.Fake.Invokes(NewRootListAction("componentstatuses", label, field), &api.ComponentStatusList{})
	if obj == nil {
		return nil, err
	}

	return obj.(*api.ComponentStatusList), err
}
