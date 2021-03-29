/*
Copyright 2019 The Kubernetes Authors.

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

package topologymanager

import (
	"k8s.io/api/core/v1"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
)

type fakeManager struct{}

//NewFakeManager returns an instance of FakeManager
func NewFakeManager() Manager {
	klog.InfoS("NewFakeManager")
	return &fakeManager{}
}

func (m *fakeManager) GetAffinity(podUID string, containerName string) TopologyHint {
	klog.InfoS("GetAffinity", "podUID", podUID, "containerName", containerName)
	return TopologyHint{}
}

func (m *fakeManager) AddHintProvider(h HintProvider) {
	klog.InfoS("AddHintProvider", "hintProvider", h)
}

func (m *fakeManager) AddContainer(pod *v1.Pod, containerID string) error {
	klog.InfoS("AddContainer", "pod", klog.KObj(pod), "containerID", containerID)
	return nil
}

func (m *fakeManager) RemoveContainer(containerID string) error {
	klog.InfoS("RemoveContainer", "containerID", containerID)
	return nil
}

func (m *fakeManager) Admit(attrs *lifecycle.PodAdmitAttributes) lifecycle.PodAdmitResult {
	klog.InfoS("Topology Admit Handler")
	return lifecycle.PodAdmitResult{
		Admit: true,
	}
}
