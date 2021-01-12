/*
Copyright 2020 The Kubernetes Authors.

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
	"fmt"

	"sync"

	"k8s.io/api/core/v1"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
)

const (
	// containerTopologyScope specifies the TopologyManagerScope per container.
	containerTopologyScope = "container"
	// podTopologyScope specifies the TopologyManagerScope per pod.
	podTopologyScope = "pod"
)

type podTopologyHints map[string]map[string]TopologyHint

// Scope interface for Topology Manager
type Scope interface {
	Name() string
	Admit(pod *v1.Pod) lifecycle.PodAdmitResult
	// AddHintProvider adds a hint provider to manager to indicate the hint provider
	// wants to be consoluted with when making topology hints
	AddHintProvider(h HintProvider)
	// AddContainer adds pod to Manager for tracking
	AddContainer(pod *v1.Pod, containerID string) error
	// RemoveContainer removes pod from Manager tracking
	RemoveContainer(containerID string) error
	// Store is the interface for storing pod topology hints
	Store
}

type scope struct {
	mutex sync.Mutex
	name  string
	// Mapping of a Pods mapping of Containers and their TopologyHints
	// Indexed by PodUID to ContainerName
	podTopologyHints podTopologyHints
	// The list of components registered with the Manager
	hintProviders []HintProvider
	// Topology Manager Policy
	policy Policy
	// Mapping of PodUID to ContainerID for Adding/Removing Pods from PodTopologyHints mapping
	podMap map[string]string
}

func (s *scope) Name() string {
	return s.name
}

func (s *scope) GetAffinity(podUID string, containerName string) TopologyHint {
	return s.podTopologyHints[podUID][containerName]
}

func (s *scope) AddHintProvider(h HintProvider) {
	s.hintProviders = append(s.hintProviders, h)
}

// It would be better to implement this function in topologymanager instead of scope
// but topologymanager do not track mapping anymore
func (s *scope) AddContainer(pod *v1.Pod, containerID string) error {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	s.podMap[containerID] = string(pod.UID)
	return nil
}

// It would be better to implement this function in topologymanager instead of scope
// but topologymanager do not track mapping anymore
func (s *scope) RemoveContainer(containerID string) error {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	klog.Infof("[topologymanager] RemoveContainer - Container ID: %v", containerID)
	podUIDString := s.podMap[containerID]
	delete(s.podMap, containerID)
	if _, exists := s.podTopologyHints[podUIDString]; exists {
		delete(s.podTopologyHints[podUIDString], containerID)
		if len(s.podTopologyHints[podUIDString]) == 0 {
			delete(s.podTopologyHints, podUIDString)
		}
	}

	return nil
}

func (s *scope) admitPolicyNone(pod *v1.Pod) lifecycle.PodAdmitResult {
	for _, container := range append(pod.Spec.InitContainers, pod.Spec.Containers...) {
		err := s.allocateAlignedResources(pod, &container)
		if err != nil {
			return unexpectedAdmissionError(err)
		}
	}
	return admitPod()
}

// It would be better to implement this function in topologymanager instead of scope
// but topologymanager do not track providers anymore
func (s *scope) allocateAlignedResources(pod *v1.Pod, container *v1.Container) error {
	for _, provider := range s.hintProviders {
		err := provider.Allocate(pod, container)
		if err != nil {
			return err
		}
	}
	return nil
}

func topologyAffinityError() lifecycle.PodAdmitResult {
	return lifecycle.PodAdmitResult{
		Message: "Resources cannot be allocated with Topology locality",
		Reason:  "TopologyAffinityError",
		Admit:   false,
	}
}

func unexpectedAdmissionError(err error) lifecycle.PodAdmitResult {
	return lifecycle.PodAdmitResult{
		Message: fmt.Sprintf("Allocate failed due to %v, which is unexpected", err),
		Reason:  "UnexpectedAdmissionError",
		Admit:   false,
	}
}

func admitPod() lifecycle.PodAdmitResult {
	return lifecycle.PodAdmitResult{Admit: true}
}
