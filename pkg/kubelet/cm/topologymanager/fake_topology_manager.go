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
	"context"

	"k8s.io/api/core/v1"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/kubelet/cm/admission"
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
)

type fakeManager struct {
	hint   *TopologyHint
	policy Policy
}

// NewFakeManager returns an instance of FakeManager
func NewFakeManager() Manager {
	// Use klog.TODO() because changing NewManager requires changes in too many other components
	logger := klog.TODO()
	logger.Info("NewFakeManager")
	return &fakeManager{}
}

// NewFakeManagerWithHint returns an instance of fake topology manager with specified topology hints
func NewFakeManagerWithHint(hint *TopologyHint) Manager {
	// Use klog.TODO() because changing NewManager requires changes in too many other components
	logger := klog.TODO()
	logger.Info("NewFakeManagerWithHint")
	return &fakeManager{
		hint:   hint,
		policy: NewNonePolicy(),
	}
}

// NewFakeManagerWithPolicy returns an instance of fake topology manager with specified policy
func NewFakeManagerWithPolicy(policy Policy) Manager {
	// Use klog.TODO() because changing NewManager requires changes in too many other components
	logger := klog.TODO()
	logger.Info("NewFakeManagerWithPolicy", "policy", policy.Name())
	return &fakeManager{
		policy: policy,
	}
}

func (m *fakeManager) GetAffinity(podUID string, containerName string) TopologyHint {
	// Use context.TODO() because we currently do not have a proper context to pass in.
	// Replace this with an appropriate context when refactoring this function to accept a context parameter.
	ctx := context.TODO()
	logger := klog.FromContext(ctx)
	logger.Info("GetAffinity", "podUID", podUID, "containerName", containerName)
	if m.hint == nil {
		return TopologyHint{}
	}

	return *m.hint
}

func (m *fakeManager) GetPolicy() Policy {
	return m.policy
}

func (m *fakeManager) AddHintProvider(logger klog.Logger, h HintProvider) {
	logger.Info("AddHintProvider", "hintProvider", h)
}

func (m *fakeManager) AddContainer(pod *v1.Pod, container *v1.Container, containerID string) {
	// Use context.TODO() because we currently do not have a proper context to pass in.
	// Replace this with an appropriate context when refactoring this function to accept a context parameter.
	ctx := context.TODO()
	logger := klog.FromContext(ctx)
	logger.Info("AddContainer", "pod", klog.KObj(pod), "containerName", container.Name, "containerID", containerID)
}

func (m *fakeManager) RemoveContainer(containerID string) error {
	// Use context.TODO() because we currently do not have a proper context to pass in.
	// Replace this with an appropriate context when refactoring this function to accept a context parameter.
	ctx := context.TODO()
	logger := klog.FromContext(ctx)
	logger.Info("RemoveContainer", "containerID", containerID)
	return nil
}

func (m *fakeManager) Admit(attrs *lifecycle.PodAdmitAttributes) lifecycle.PodAdmitResult {
	// TODO: create context here as changing interface https://github.com/kubernetes/kubernetes/blob/09aaf7226056a7964adcb176d789de5507313d00/pkg/kubelet/lifecycle/interfaces.go#L43
	// requires changes in too many other components
	ctx := context.TODO()
	logger := klog.FromContext(ctx)
	logger.Info("Topology Admit Handler")
	return admission.GetPodAdmitResult(nil)
}
