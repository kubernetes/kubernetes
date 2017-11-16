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
	"k8s.io/api/core/v1"
	policy "k8s.io/api/policy/v1beta1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/kubernetes/plugin/pkg/scheduler/schedulercache"
)

// FakeCache is used for testing
type FakeCache struct {
	AssumeFunc       func(*v1.Pod)
	ForgetFunc       func(*v1.Pod)
	IsAssumedPodFunc func(*v1.Pod) bool
	GetPodFunc       func(*v1.Pod) *v1.Pod
}

func (f *FakeCache) AssumePod(pod *v1.Pod) error {
	f.AssumeFunc(pod)
	return nil
}

func (f *FakeCache) FinishBinding(pod *v1.Pod) error { return nil }

func (f *FakeCache) ForgetPod(pod *v1.Pod) error {
	f.ForgetFunc(pod)
	return nil
}

func (f *FakeCache) AddPod(pod *v1.Pod) error { return nil }

func (f *FakeCache) UpdatePod(oldPod, newPod *v1.Pod) error { return nil }

func (f *FakeCache) RemovePod(pod *v1.Pod) error { return nil }

func (f *FakeCache) IsAssumedPod(pod *v1.Pod) (bool, error) {
	return f.IsAssumedPodFunc(pod), nil
}

func (f *FakeCache) GetPod(pod *v1.Pod) (*v1.Pod, error) {
	return f.GetPodFunc(pod), nil
}

func (f *FakeCache) AddNode(node *v1.Node) error { return nil }

func (f *FakeCache) UpdateNode(oldNode, newNode *v1.Node) error { return nil }

func (f *FakeCache) RemoveNode(node *v1.Node) error { return nil }

func (f *FakeCache) UpdateNodeNameToInfoMap(infoMap map[string]*schedulercache.NodeInfo) error {
	return nil
}

func (f *FakeCache) AddPDB(pdb *policy.PodDisruptionBudget) error { return nil }

func (f *FakeCache) UpdatePDB(oldPDB, newPDB *policy.PodDisruptionBudget) error { return nil }

func (f *FakeCache) RemovePDB(pdb *policy.PodDisruptionBudget) error { return nil }

func (f *FakeCache) ListPDBs(selector labels.Selector) ([]*policy.PodDisruptionBudget, error) {
	return nil, nil
}

func (f *FakeCache) List(s labels.Selector) ([]*v1.Pod, error) { return nil, nil }

func (f *FakeCache) FilteredList(filter schedulercache.PodFilter, selector labels.Selector) ([]*v1.Pod, error) {
	return nil, nil
}
