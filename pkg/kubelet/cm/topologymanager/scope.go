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
	"context"
	"errors"
	"fmt"
	"sync"

	"k8s.io/api/core/v1"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/kubelet/cm/admission"
	"k8s.io/kubernetes/pkg/kubelet/cm/containermap"
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
)

const (
	// ContainerTopologyScope specifies the TopologyManagerScope per container.
	ContainerTopologyScope = "container"
	// PodTopologyScope specifies the TopologyManagerScope per pod.
	PodTopologyScope = "pod"
	// noneTopologyScope specifies the TopologyManagerScope when topologyPolicyName is none.
	NoneTopologyScope = "none"
)

type podTopologyHints map[string]map[string]TopologyHint

// Scope interface for Topology Manager
type Scope interface {
	Name() string
	GetPolicy() Policy
	Admit(ctx context.Context, pod *v1.Pod, operation lifecycle.Operation) lifecycle.PodAdmitResult
	// AddHintProvider adds a hint provider to manager to indicate the hint provider
	// wants to be consoluted with when making topology hints
	AddHintProvider(logger klog.Logger, h HintProvider)
	// AddContainer adds pod to Manager for tracking
	AddContainer(pod *v1.Pod, container *v1.Container, containerID string)
	// RemoveContainer removes pod from Manager tracking
	RemoveContainer(logger klog.Logger, containerID string) error
	// Store is the interface for storing pod topology hints
	Store
	// Scope forwards the admission transaction to the hint providers, so it has
	// the methods of AdmitTransactionalHintProvider.
	AdmitTransactionalHintProvider
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
	// Mapping of (PodUid, ContainerName) to ContainerID for Adding/Removing Pods from PodTopologyHints mapping
	podMap containermap.ContainerMap
}

func (s *scope) Name() string {
	return s.name
}

func (s *scope) getTopologyHints(podUID string, containerName string) TopologyHint {
	s.mutex.Lock()
	defer s.mutex.Unlock()
	return s.podTopologyHints[podUID][containerName]
}

func (s *scope) setTopologyHints(podUID string, containerName string, th TopologyHint) {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	if s.podTopologyHints[podUID] == nil {
		s.podTopologyHints[podUID] = make(map[string]TopologyHint)
	}
	s.podTopologyHints[podUID][containerName] = th
}

func (s *scope) GetAffinity(_ klog.Logger, podUID string, containerName string) TopologyHint {
	return s.getTopologyHints(podUID, containerName)
}

func (s *scope) GetPolicy() Policy {
	return s.policy
}

func (s *scope) AddHintProvider(_ klog.Logger, h HintProvider) {
	s.hintProviders = append(s.hintProviders, h)
}

// It would be better to implement this function in topologymanager instead of scope
// but topologymanager do not track mapping anymore
func (s *scope) AddContainer(pod *v1.Pod, container *v1.Container, containerID string) {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	s.podMap.Add(string(pod.UID), container.Name, containerID)
}

// It would be better to implement this function in topologymanager instead of scope
// but topologymanager do not track mapping anymore
func (s *scope) RemoveContainer(logger klog.Logger, containerID string) error {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	logger.Info("RemoveContainer", "containerID", containerID)
	// Get the podUID and containerName associated with the containerID to be removed and remove it
	podUIDString, containerName, err := s.podMap.GetContainerRef(containerID)
	if err != nil {
		return nil
	}
	s.podMap.RemoveByContainerID(containerID)

	// In cases where a container has been restarted, it's possible that the same podUID and
	// containerName are already associated with a *different* containerID now. Only remove
	// the TopologyHints associated with that podUID and containerName if this is not true
	if _, err := s.podMap.GetContainerID(podUIDString, containerName); err != nil {
		delete(s.podTopologyHints[podUIDString], containerName)
		if len(s.podTopologyHints[podUIDString]) == 0 {
			delete(s.podTopologyHints, podUIDString)
		}
	}

	return nil
}

func (s *scope) admitPolicyNone(ctx context.Context, pod *v1.Pod, operation lifecycle.Operation) lifecycle.PodAdmitResult {
	for _, container := range append(pod.Spec.InitContainers, pod.Spec.Containers...) {
		err := s.allocateAlignedResources(ctx, pod, &container, operation)
		if err != nil {
			return admission.GetPodAdmitResult(err)
		}
	}
	return admission.GetPodAdmitResult(nil)
}

// It would be better to implement this function in topologymanager instead of scope
// but topologymanager does not track providers anymore
func (s *scope) allocateAlignedResources(ctx context.Context, pod *v1.Pod, container *v1.Container, operation lifecycle.Operation) error {
	for _, provider := range s.hintProviders {
		err := provider.Allocate(ctx, pod, container, operation)
		if err != nil {
			return err
		}
	}
	return nil
}

// StageAdmit stages the admission of the pod in every hint provider that
// implements AdmitTransactionalHintProvider, stopping at the first failure.
func (s *scope) StageAdmit(ctx context.Context, pod *v1.Pod, operation lifecycle.Operation) error {
	for _, provider := range s.hintProviders {
		if tx, ok := provider.(AdmitTransactionalHintProvider); ok {
			if err := tx.StageAdmit(ctx, pod, operation); err != nil {
				return fmt.Errorf("hint provider %T failed to stage the %s of pod %s: %w", provider, operation, klog.KObj(pod), err)
			}
		}
	}
	return nil
}

// CommitAdmit commits the admission of the pod in every hint provider that
// implements AdmitTransactionalHintProvider. A failing provider does not stop the others,
// because the admission is accepted already.
func (s *scope) CommitAdmit(ctx context.Context, pod *v1.Pod, operation lifecycle.Operation) error {
	var errs []error
	for _, provider := range s.hintProviders {
		if tx, ok := provider.(AdmitTransactionalHintProvider); ok {
			if err := tx.CommitAdmit(ctx, pod, operation); err != nil {
				errs = append(errs, fmt.Errorf("hint provider %T failed to commit the %s of pod %s: %w", provider, operation, klog.KObj(pod), err))
			}
		}
	}
	return errors.Join(errs...)
}

// AbortAdmit aborts the admission of the pod in every hint provider that
// implements AdmitTransactionalHintProvider.
// Hints written by an aborted admission are not restored: they are only read
// by Allocate and AllocatePod, after the next admission of the pod has
// rewritten them.
func (s *scope) AbortAdmit(ctx context.Context, pod *v1.Pod, operation lifecycle.Operation) {
	for _, provider := range s.hintProviders {
		if tx, ok := provider.(AdmitTransactionalHintProvider); ok {
			tx.AbortAdmit(ctx, pod, operation)
		}
	}
}

func (s *scope) allocatePodAlignedResources(logger klog.Logger, pod *v1.Pod, operation lifecycle.Operation) error {
	for _, provider := range s.hintProviders {
		err := provider.AllocatePod(logger, pod, operation)
		if err != nil {
			return err
		}
	}
	return nil
}
