/*
Copyright 2025 The Kubernetes Authors.

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

package userns

import (
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
)

type UsernsManager struct{}

func MakeUserNsManager(kl userNsPodsManager) (*UsernsManager, error) {
	return nil, nil
}

// Release releases the user namespace allocated to the specified pod.
func (m *UsernsManager) Release(podUID types.UID) {
	return
}

func (m *UsernsManager) GetOrCreateUserNamespaceMappings(pod *v1.Pod, runtimeHandler string) (*runtimeapi.UserNamespace, error) {
	return nil, nil
}

// CleanupOrphanedPodUsernsAllocations reconciliates the state of user namespace
// allocations with the pods actually running. It frees any user namespace
// allocation for orphaned pods.
func (m *UsernsManager) CleanupOrphanedPodUsernsAllocations(pods []*v1.Pod, runningPods []*kubecontainer.Pod) error {
	return nil
}

func EnabledUserNamespacesSupport() bool {
	return false
}
