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

	v1 "k8s.io/api/core/v1"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/kubelet/cm/admission"
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
)

type fakeManager struct {
	hint   *TopologyHint
	policy Policy
	scope  string
}

// NewFakeManager returns an instance of FakeManager
func NewFakeManager(logger klog.Logger) Manager {
	logger.Info("NewFakeManager")
	return &fakeManager{}
}

// NewFakeManagerWithScope returns an instance of fake topology manager with specified scope
func NewFakeManagerWithScope(scope string) Manager {
	return &fakeManager{
		scope: scope,
	}
}

// NewFakeManagerWithHint returns an instance of fake topology manager with specified topology hints
func NewFakeManagerWithHint(logger klog.Logger, hint *TopologyHint) Manager {
	logger.Info("NewFakeManagerWithHint")
	return &fakeManager{
		hint:   hint,
		policy: NewNonePolicy(),
	}
}

// NewFakeManagerWithPolicy returns an instance of fake topology manager with specified policy
func NewFakeManagerWithPolicy(logger klog.Logger, policy Policy) Manager {
	logger.Info("NewFakeManagerWithPolicy", "policy", policy.Name())
	return &fakeManager{
		policy: policy,
	}
}

func (m *fakeManager) GetAffinity(logger klog.Logger, podUID string, containerName string) TopologyHint {
	logger.Info("GetAffinity", "podUID", podUID, "containerName", containerName)
	if m.hint == nil {
		return TopologyHint{}
	}

	return *m.hint
}

func (m *fakeManager) GetPolicy() Policy {
	return m.policy
}

func (m *fakeManager) Name() string {
	return m.scope
}

func (m *fakeManager) AddHintProvider(logger klog.Logger, h HintProvider) {
	logger.Info("AddHintProvider", "hintProvider", h)
}

func (m *fakeManager) AddContainer(logger klog.Logger, pod *v1.Pod, container *v1.Container, containerID string) {
	logger.Info("AddContainer", "pod", klog.KObj(pod), "containerName", container.Name, "containerID", containerID)
}

func (m *fakeManager) RemoveContainer(logger klog.Logger, containerID string) error {
	logger.Info("RemoveContainer", "containerID", containerID)
	return nil
}

func (m *fakeManager) Admit(ctx context.Context, attrs *lifecycle.PodAdmitAttributes) lifecycle.PodAdmitResult {
	logger := klog.FromContext(ctx)
	logger.Info("Topology Admit Handler")
	return admission.GetPodAdmitResult(nil)
}
