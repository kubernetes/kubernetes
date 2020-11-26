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

package kubelet

import (
	"sort"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1alpha2"
	"k8s.io/klog/v2"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
)

const (
	// The number of sandboxes which can be deleted in parallel.
	sandboxDeletionBufferLimit = 20
)

type sandboxStatusByCreatedList []*runtimeapi.PodSandboxStatus

type podSandboxDeleter struct {
	worker chan<- string
}

func (a sandboxStatusByCreatedList) Len() int      { return len(a) }
func (a sandboxStatusByCreatedList) Swap(i, j int) { a[i], a[j] = a[j], a[i] }
func (a sandboxStatusByCreatedList) Less(i, j int) bool {
	return a[i].CreatedAt > a[j].CreatedAt
}

func newPodSandboxDeleter(runtime kubecontainer.Runtime) *podSandboxDeleter {
	buffer := make(chan string, sandboxDeletionBufferLimit)
	go wait.Forever(func() {
		for id := range buffer {
			if err := runtime.DeleteSandbox(id); err != nil {
				klog.Warningf("[pod_sandbox_deleter] DeleteSandbox returned error for (id=%v): %v", id, err)
			}
		}
	}, 0)

	return &podSandboxDeleter{
		worker: buffer,
	}
}

// deleteSandboxesInPod issues sandbox deletion requests for all inactive sandboxes after sorting by creation time
// and skipping toKeep number of sandboxes
func (p *podSandboxDeleter) deleteSandboxesInPod(podStatus *kubecontainer.PodStatus, toKeep int) {
	sandboxIDs := sets.NewString()
	for _, containerStatus := range podStatus.ContainerStatuses {
		sandboxIDs.Insert(containerStatus.PodSandboxID)
	}
	sandboxStatuses := podStatus.SandboxStatuses
	if toKeep > 0 {
		sort.Sort(sandboxStatusByCreatedList(sandboxStatuses))
	}

	for i := len(sandboxStatuses) - 1; i >= toKeep; i-- {
		if _, ok := sandboxIDs[sandboxStatuses[i].Id]; !ok && sandboxStatuses[i].State != runtimeapi.PodSandboxState_SANDBOX_READY {
			select {
			case p.worker <- sandboxStatuses[i].Id:
			default:
				klog.Warningf("Failed to issue the request to remove sandbox %v", sandboxStatuses[i].Id)
			}
		}
	}
}
