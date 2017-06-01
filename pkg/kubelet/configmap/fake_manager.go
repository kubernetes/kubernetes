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

package configmap

import (
	"k8s.io/kubernetes/pkg/api/v1"
)

// fakeManager implements Manager interface for testing purposes.
// simple operations to apiserver.
type fakeManager struct {
}

func NewFakeManager() Manager {
	return &fakeManager{}
}

func (s *fakeManager) GetConfigMap(namespace, name string) (*v1.ConfigMap, error) {
	return nil, nil
}

func (s *fakeManager) RegisterPod(pod *v1.Pod) {
}

func (s *fakeManager) UnregisterPod(pod *v1.Pod) {
}
