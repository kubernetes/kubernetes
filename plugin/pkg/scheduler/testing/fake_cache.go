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

package testing

import (
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/plugin/pkg/scheduler/schedulercache"
)

// FakeCache is used for testing
type FakeCache struct {
	AssumeFunc func(*v1.Pod)
}

func (f *FakeCache) AssumePod(pod *v1.Pod) error {
	f.AssumeFunc(pod)
	return nil
}

func (f *FakeCache) FinishBinding(pod *v1.Pod) error { return nil }

func (f *FakeCache) ForgetPod(pod *v1.Pod) error { return nil }

func (f *FakeCache) AddPod(pod *v1.Pod) error { return nil }

func (f *FakeCache) UpdatePod(oldPod, newPod *v1.Pod) error { return nil }

func (f *FakeCache) RemovePod(pod *v1.Pod) error { return nil }

func (f *FakeCache) AddNode(node *v1.Node) error { return nil }

func (f *FakeCache) UpdateNode(oldNode, newNode *v1.Node) error { return nil }

func (f *FakeCache) RemoveNode(node *v1.Node) error { return nil }

func (f *FakeCache) UpdateNodeNameToInfoMap(infoMap map[string]*schedulercache.NodeInfo) error {
	return nil
}

func (f *FakeCache) List(s labels.Selector) ([]*v1.Pod, error) { return nil, nil }
