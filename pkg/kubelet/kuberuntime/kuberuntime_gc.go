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
)

type containerGCInfo struct {
	containerID string
	createdAt   time.Time
}

type sandboxGCInfo struct {
	sandboxID  string
	createdAt  time.Time
	containers []*containerGCInfo
	client     internalApi.RuntimeService
}

func (gc *sandboxGCInfo) containersNum() int {
	return len(gc.containers)
}

func (gc *sandboxGCInfo) removeOldestNContainers(n int) {
	if n <= 0 {
		return
	}

	toKeep := len(gc.containers) - n
	for i := toKeep; i < len(gc.containers); i++ {
		err := gc.client.RemoveContainer(gc.containers[i].containerID)
		if err != nil {
			glog.Warningf("Failed to remove dead container %s: %s", gc.containers[i].containerID, err)
		}
	}

	if toKeep == 0 {
		// Remove sandbox with 0 containers
		err := gc.client.RemovePodSandbox(gc.sandboxID)
		if err != nil {
			glog.Warningf("Failed to remove dead pod sandbox %s: %s", gc.sandboxID, err)
		}
	}

	gc.containers = gc.containers[:toKeep]
}

// Newest first.
type containerByCreated []*containerGCInfo

func (a containerByCreated) Len() int           { return len(a) }
func (a containerByCreated) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a containerByCreated) Less(i, j int) bool { return a[i].createdAt.After(a[j].createdAt) }

// Newest first.
type sandboxByCreated []*sandboxGCInfo

func (a sandboxByCreated) Len() int           { return len(a) }
func (a sandboxByCreated) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a sandboxByCreated) Less(i, j int) bool { return a[i].createdAt.After(a[j].createdAt) }

// evictableSandboxes gets evictable pod sandboxes.
// Evictable pod sandboxes are not active and created more than MinAge ago)
func (m *kubeGenericRuntimeManager) evictableSandboxes(minAge time.Duration) ([]*sandboxGCInfo, error) {
	sandboxes, err := m.getKubeletSandboxes(true)
	if err != nil {
		return nil, err
	}

	evictableSandboxes := make([]*sandboxGCInfo, 0)
	newestGCTime := time.Now().Add(-minAge)
	for _, s := range sandboxes {
		if s.GetState() != runtimeApi.PodSandBoxState_NOTREADY {
			continue
		}

		createdAt := time.Unix(*s.CreatedAt, 0)
		if newestGCTime.Before(createdAt) {
			continue
		}

		sandboxGCInfo := &sandboxGCInfo{
			sandboxID:  s.GetId(),
			createdAt:  createdAt,
			containers: make([]*containerGCInfo, 0),
			client:     m.runtimeService,
		}
		containerStatuses, err := m.getKubeletContainerStatuses(s.GetId())
		if err != nil {
			return nil, err
		}
		for _, c := range containerStatuses {
			sandboxGCInfo.containers = append(sandboxGCInfo.containers, &containerGCInfo{
				containerID: c.ID.ID,
				createdAt:   c.CreatedAt,
			})
		}

		sort.Sort(containerByCreated(sandboxGCInfo.containers))
		evictableSandboxes = append(evictableSandboxes, sandboxGCInfo)
	}

	sort.Sort(sandboxByCreated(evictableSandboxes))

	return evictableSandboxes, nil
}

// GarbageCollect removes dead containers using the specified container gc policy
func (m *kubeGenericRuntimeManager) GarbageCollect(gcPolicy kubecontainer.ContainerGCPolicy, allSourcesReady bool) error {
	evictablSandboxes, err := m.evictableSandboxes(gcPolicy.MinAge)
	if err != nil {
		glog.Warningf("Make evictableSandboxes failed: %v", err)
		return err
	}

	glog.V(4).Infof("GarbageCollect gets sandboxes %q for evicting", evictablSandboxes)

	// Enforce max number of dead containers of each pod sandbox
	totalEvictableContainers := 0
	for _, s := range evictablSandboxes {
		if gcPolicy.MaxPerPodContainer > 0 {
			s.removeOldestNContainers(s.containersNum() - gcPolicy.MaxPerPodContainer)
		}
		totalEvictableContainers += s.containersNum()
	}

	// Enforce max number of containers
	if gcPolicy.MaxContainers > 0 && totalEvictableContainers > gcPolicy.MaxContainers {
		containerNumToRemove := totalEvictableContainers - gcPolicy.MaxContainers
		sandboxIndex := len(evictablSandboxes) - 1

		// remove oldest first
		for containerNumToRemove > 0 && sandboxIndex >= 0 {
			s := evictablSandboxes[sandboxIndex]
			removedNum := s.containersNum()
			s.removeOldestNContainers(removedNum)

			containerNumToRemove -= removedNum
			sandboxIndex--
		}
	}

	return nil
}
