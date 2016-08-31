/*
Copyright 2016 The Kubernetes Authors.

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

package kuberuntime

import (
	"sort"
	"time"

	"github.com/golang/glog"
	internalApi "k8s.io/kubernetes/pkg/kubelet/api"
	runtimeApi "k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/runtime"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/types"
)

const (
	// nonKubernetesPodUID is used to track evictable sandboxes/containers not created by kubelet.
	nonKubernetesPodUID = "nonKubernetesPodUID"
)

// containerGCInfo contains minimal information for container GC.
type containerGCInfo struct {
	podUID      string
	containerID string
	sandboxID   string
	createdAt   time.Time
}

// podGCInfo contains minimal information for pod GC.
type podGCInfo struct {
	sandboxIDs []string
	containers []*containerGCInfo
	client     internalApi.RuntimeService
}

// containersNum returns the number of containers belonging to the pod.
func (gc *podGCInfo) containersNum() int {
	return len(gc.containers)
}

// sandboxContainersCount returns the count of containers belonging to the sandbox.
func (gc *podGCInfo) sandboxContainersCount(sandboxID string) int {
	count := 0
	for _, c := range gc.containers {
		if c.sandboxID == sandboxID {
			count++
		}
	}

	return count
}

// removeEvictableSandboxes removes dead sandboxes which associate with 0 containers.
func (gc *podGCInfo) removeEvictableSandboxes() {
	for _, sandboxID := range gc.sandboxIDs {
		if gc.sandboxContainersCount(sandboxID) > 0 {
			continue
		}

		if err := gc.client.RemovePodSandbox(sandboxID); err != nil {
			glog.Warningf("Failed to remove dead sandbox %q: %v", sandboxID, err)
		}
	}
}

// removeContainer removes the container by ID and returns its index in gc.containers.
// Note that it doesn't delete the container from gc.containers.
func (gc *podGCInfo) removeContainer(containerID string) int {
	containerIndex := -1
	for i, c := range gc.containers {
		if containerID == c.containerID {
			containerIndex = i
			break
		}
	}

	if containerIndex != -1 {
		if err := gc.client.RemoveContainer(containerID); err != nil {
			glog.Warningf("Failed to remove dead container %q: %v", containerID, err)
		}
	}

	return containerIndex
}

// removeOldestNContainers removes oldest n containers.
func (gc *podGCInfo) removeOldestNContainers(n int) {
	if n <= 0 {
		return
	}

	toKeep := len(gc.containers) - n
	for i := toKeep; i < len(gc.containers); i++ {
		gc.removeContainer(gc.containers[i].containerID)
	}

	gc.containers = gc.containers[:toKeep]
}

// Newest first.
type containerByCreated []*containerGCInfo

func (a containerByCreated) Len() int           { return len(a) }
func (a containerByCreated) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a containerByCreated) Less(i, j int) bool { return a[i].createdAt.After(a[j].createdAt) }

// toContainerGCInfo converts kubecontainer.ContainerStatus to containerGCInfo.
func toContainerGCInfo(podUID, sandboxID string, c *kubecontainer.ContainerStatus) *containerGCInfo {
	return &containerGCInfo{
		containerID: c.ID.ID,
		createdAt:   c.CreatedAt,
		sandboxID:   sandboxID,
		podUID:      podUID,
	}
}

// getEvictablePods gets evictable pods.
// Evictable pods contain sandboxes and containers which are not active and created more than MinAge ago.
func (m *kubeGenericRuntimeManager) getEvictablePods(minAge time.Duration) (map[string]*podGCInfo, error) {
	sandboxes, err := m.getKubeletSandboxes(true)
	if err != nil {
		return nil, err
	}

	evictablePods := make(map[string]*podGCInfo)
	newestGCTime := time.Now().Add(-minAge)
	for _, s := range sandboxes {
		if s.GetState() != runtimeApi.PodSandBoxState_NOTREADY {
			continue
		}

		createdAt := time.Unix(*s.CreatedAt, 0)
		if newestGCTime.Before(createdAt) {
			continue
		}

		podUID := getStringValueFromLabel(s.Labels, types.KubernetesPodUIDLabel)
		if len(podUID) == 0 {
			podUID = nonKubernetesPodUID
		}
		if pod, ok := evictablePods[podUID]; ok {
			pod.sandboxIDs = append(pod.sandboxIDs, s.GetId())
		} else {
			evictablePods[podUID] = &podGCInfo{
				client:     m.runtimeService,
				containers: make([]*containerGCInfo, 0),
				sandboxIDs: []string{s.GetId()},
			}
		}

		containerStatuses, err := m.getKubeletContainerStatuses(s.GetId())
		if err != nil {
			return nil, err
		}
		for _, c := range containerStatuses {
			evictablePods[podUID].containers = append(evictablePods[podUID].containers,
				toContainerGCInfo(podUID, s.GetId(), c))
		}

		sort.Sort(containerByCreated(evictablePods[podUID].containers))
	}

	return evictablePods, nil
}

// enforceMaxPerPodContainer enforces max number of dead containers of each pod and
// returns the number of total evictable containers.
func enforceMaxPerPodContainer(evictablePods map[string]*podGCInfo, maxPerPodContainer int) int {
	totalEvictableContainers := 0
	for _, pod := range evictablePods {
		if maxPerPodContainer > 0 {
			pod.removeOldestNContainers(pod.containersNum() - maxPerPodContainer)
		}
		totalEvictableContainers += pod.containersNum()
	}

	return totalEvictableContainers
}

// GarbageCollect removes dead containers using the specified container gc policy.
// It consists of following steps:
// * gets evictable pods with belonging containers and sandboxes which are not active and created more than gcPolicy.MinAge ago.
// * removes oldest dead containers for each pod by enforcing gcPolicy.MaxPerPodContainer.
// * removes oldest dead containers by enforcing gcPolicy.MaxContainers.
// * removes dead sandboxes with zero containers.
func (m *kubeGenericRuntimeManager) GarbageCollect(gcPolicy kubecontainer.ContainerGCPolicy, allSourcesReady bool) error {
	evictablePods, err := m.getEvictablePods(gcPolicy.MinAge)
	if err != nil {
		glog.Warningf("getEvictablePods failed: %v", err)
		return err
	}

	glog.V(4).Infof("GarbageCollect gets pods %q for evicting", evictablePods)

	// Enforce max number of dead containers of each pod
	totalEvictableContainers := enforceMaxPerPodContainer(evictablePods, gcPolicy.MaxPerPodContainer)
	// Enforce max number of containers
	if gcPolicy.MaxContainers > 0 && totalEvictableContainers > gcPolicy.MaxContainers {
		// sort all evitable containers by createdAt (newest first)
		allEvictableContainers := make([]*containerGCInfo, 0)
		for _, pod := range evictablePods {
			allEvictableContainers = append(allEvictableContainers, pod.containers...)
		}
		sort.Sort(containerByCreated(allEvictableContainers))

		// Remove oldest (totalEvictableContainers - gcPolicy.MaxContainers) containers
		for i := gcPolicy.MaxContainers; i < totalEvictableContainers; i++ {
			container := allEvictableContainers[i]
			containerList := evictablePods[container.podUID].containers
			// remove the container from runtime service and also update podGCInfo.containers
			containerIndex := evictablePods[container.podUID].removeContainer(container.containerID)
			evictablePods[container.podUID].containers = append(containerList[:containerIndex], containerList[containerIndex+1:]...)
		}
	}

	// Remove sandboxes with zero containers
	for _, pod := range evictablePods {
		pod.removeEvictableSandboxes()
	}

	return nil
}
