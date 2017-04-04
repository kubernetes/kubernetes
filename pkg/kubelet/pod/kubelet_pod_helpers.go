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

package pod

import (
	"k8s.io/kubernetes/pkg/api/v1"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/qos"
	kubelettypes "k8s.io/kubernetes/pkg/kubelet/types"
)

func (p *Pod) IsMirrorPod() bool {
	return IsMirrorPod(p.apiPod)
}

func (p *Pod) IsStatic() bool {
	return IsStaticPod(p.apiPod)
}

func (p *Pod) IsCritical() bool {
	return kubelettypes.IsCriticalPod(p.apiPod)
}

// notRunning returns true if every status is terminated or waiting, or the status list
// is empty.
func notRunning(statuses []v1.ContainerStatus) bool {
	for _, status := range statuses {
		if status.State.Terminated == nil && status.State.Waiting == nil {
			return false
		}
	}
	return true
}

// IsMirrorPodOf returns true if mirrorPod is a correct representation of
// pod; false otherwise.
func (p *Pod) IsMirrorPodOf(pod *Pod) bool {
	// Check name and namespace first.
	if pod.apiPod.Name != p.apiPod.Name || pod.apiPod.Namespace != p.apiPod.Namespace {
		return false
	}
	hash, ok := getHashFromMirrorPod(p.apiPod)
	if !ok {
		return false
	}
	return hash == getPodHash(pod.apiPod)
}

func (p *Pod) GetPodQOS() v1.PodQOSClass {
	return qos.GetPodQOS(p.GetSpec())
}

func (p *Pod) IsHostNetworkPod() bool {
	return kubecontainer.IsHostNetworkPod(p.GetAPIPod())
}

func (p *Pod) ShouldContainerBeRestarted(container *v1.Container, podStatus *kubecontainer.PodStatus) bool {
	return kubecontainer.ShouldContainerBeRestarted(container, p.GetAPIPod(), podStatus)
}

func (p *Pod) GetContainerSpec(containerName string) *v1.Container {
	return kubecontainer.GetContainerSpec(p.GetAPIPod(), containerName)
}
