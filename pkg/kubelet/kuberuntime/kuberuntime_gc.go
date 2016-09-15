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
	"k8s.io/kubernetes/pkg/types"
)

// containerGC is the manager of garbage collection.
type containerGC struct {
	client    internalApi.RuntimeService
	manager   *kubeGenericRuntimeManager
	podGetter podGetter
}

// NewContainerGC creates a new containerGC.
func NewContainerGC(client internalApi.RuntimeService, podGetter podGetter, manager *kubeGenericRuntimeManager) *containerGC {
	return &containerGC{
		client:    client,
		manager:   manager,
		podGetter: podGetter,
	}
}

// containerGCInfo is the internal information kept for containers being considered for GC.
type containerGCInfo struct {
	// The ID of the container.
	id string
	// The name of the container.
	name string
	// The sandbox ID which this container belongs to
	sandboxID string
	// Creation time for the container.
	createTime time.Time
}

// evictUnit is considered for eviction as units of (UID, container name) pair.
type evictUnit struct {
	// UID of the pod.
	uid types.UID
	// Name of the container in the pod.
	name string
}

type containersByEvictUnit map[evictUnit][]containerGCInfo

// NumContainers returns the number of containers in this map.
func (cu containersByEvictUnit) NumContainers() int {
	num := 0
	for key := range cu {
		num += len(cu[key])
	}
	return num
}

// NumEvictUnits returns the number of pod in this map.
func (cu containersByEvictUnit) NumEvictUnits() int {
	return len(cu)
}

// Newest first.
type byCreated []containerGCInfo

func (a byCreated) Len() int           { return len(a) }
func (a byCreated) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a byCreated) Less(i, j int) bool { return a[i].createTime.After(a[j].createTime) }

// enforceMaxContainersPerEvictUnit enforces MaxPerPodContainer for each evictUnit.
func (cgc *containerGC) enforceMaxContainersPerEvictUnit(evictUnits containersByEvictUnit, MaxContainers int) {
	for key := range evictUnits {
		toRemove := len(evictUnits[key]) - MaxContainers

		if toRemove > 0 {
			evictUnits[key] = cgc.removeOldestN(evictUnits[key], toRemove)
		}
	}
}

// removeOldestN removes the oldest toRemove containers and returns the resulting slice.
func (cgc *containerGC) removeOldestN(containers []containerGCInfo, toRemove int) []containerGCInfo {
	// Remove from oldest to newest (last to first).
	numToKeep := len(containers) - toRemove
	for i := numToKeep; i < len(containers); i++ {
		cgc.removeContainer(containers[i].id, containers[i].name)
	}

	// Assume we removed the containers so that we're not too aggressive.
	return containers[:numToKeep]
}

// removeContainer removes the container by containerID.
func (cgc *containerGC) removeContainer(containerID, containerName string) {
	glog.V(4).Infof("Removing container %q name %q", containerID, containerName)
	if err := cgc.client.RemoveContainer(containerID); err != nil {
		glog.Warningf("Failed to remove container %q: %v", containerID, err)
	}
}

// removeSandbox removes the sandbox by sandboxID.
func (cgc *containerGC) removeSandbox(sandboxID string) {
	glog.V(4).Infof("Removing sandbox %q", sandboxID)
	if err := cgc.client.RemovePodSandbox(sandboxID); err != nil {
		glog.Warningf("Failed to remove sandbox %q: %v", sandboxID, err)
	}
}

// evictableContainers gets all containers that are evictable. Evictable containers are: not running
// and created more than MinAge ago.
func (cgc *containerGC) evictableContainers(minAge time.Duration) (containersByEvictUnit, error) {
	containers, err := cgc.manager.getKubeletContainers(true)
	if err != nil {
		return containersByEvictUnit{}, err
	}

	evictUnits := make(containersByEvictUnit)
	newestGCTime := time.Now().Add(-minAge)
	for _, container := range containers {
		// Prune out running containers.
		if container.GetState() == runtimeApi.ContainerState_RUNNING {
			continue
		}

		createdAt := time.Unix(container.GetCreatedAt(), 0)
		if newestGCTime.Before(createdAt) {
			continue
		}

		labeledInfo := getContainerInfoFromLabels(container.Labels)
		containerInfo := containerGCInfo{
			id:         container.GetId(),
			name:       container.Metadata.GetName(),
			createTime: createdAt,
			sandboxID:  container.GetPodSandboxId(),
		}
		key := evictUnit{
			uid:  labeledInfo.PodUID,
			name: containerInfo.name,
		}
		evictUnits[key] = append(evictUnits[key], containerInfo)
	}

	// Sort the containers by age.
	for uid := range evictUnits {
		sort.Sort(byCreated(evictUnits[uid]))
	}

	return evictUnits, nil
}

// evictableSandboxes gets all sandboxes that are evictable. Evictable sandboxes are: not running
// and contains no containers at all.
func (cgc *containerGC) evictableSandboxes() ([]string, error) {
	containers, err := cgc.manager.getKubeletContainers(true)
	if err != nil {
		return nil, err
	}

	sandboxes, err := cgc.manager.getKubeletSandboxes(true)
	if err != nil {
		return nil, err
	}

	evictSandboxes := make([]string, 0)
	for _, sandbox := range sandboxes {
		// Prune out ready sandboxes.
		if sandbox.GetState() == runtimeApi.PodSandBoxState_READY {
			continue
		}

		// Prune out sandboxes that still have containers.
		found := false
		sandboxID := sandbox.GetId()
		for _, container := range containers {
			if container.GetPodSandboxId() == sandboxID {
				found = true
				break
			}
		}
		if found {
			continue
		}

		evictSandboxes = append(evictSandboxes, sandboxID)
	}

	return evictSandboxes, nil
}

// isPodDeleted returns true if the pod is already deleted.
func (cgc *containerGC) isPodDeleted(podUID types.UID) bool {
	_, found := cgc.podGetter.GetPodByUID(podUID)
	return !found
}

// GarbageCollect removes dead containers using the specified container gc policy.
// Note that gc policy is not applied to sandboxes. Sandboxes are only removed when they are
// not ready and containing no containers.
//
// GarbageCollect consists of the following steps:
// * gets evictable containers which are not active and created more than gcPolicy.MinAge ago.
// * removes oldest dead containers for each pod by enforcing gcPolicy.MaxPerPodContainer.
// * removes oldest dead containers by enforcing gcPolicy.MaxContainers.
// * gets evictable sandboxes which are not ready and contains no containers.
// * removes evictable sandboxes.
func (cgc *containerGC) GarbageCollect(gcPolicy kubecontainer.ContainerGCPolicy, allSourcesReady bool) error {
	// Separate containers by evict units.
	evictUnits, err := cgc.evictableContainers(gcPolicy.MinAge)
	if err != nil {
		return err
	}

	// Remove deleted pod containers if all sources are ready.
	if allSourcesReady {
		for key, unit := range evictUnits {
			if cgc.isPodDeleted(key.uid) {
				cgc.removeOldestN(unit, len(unit)) // Remove all.
				delete(evictUnits, key)
			}
		}
	}

	// Enforce max containers per evict unit.
	if gcPolicy.MaxPerPodContainer >= 0 {
		cgc.enforceMaxContainersPerEvictUnit(evictUnits, gcPolicy.MaxPerPodContainer)
	}

	// Enforce max total number of containers.
	if gcPolicy.MaxContainers >= 0 && evictUnits.NumContainers() > gcPolicy.MaxContainers {
		// Leave an equal number of containers per evict unit (min: 1).
		numContainersPerEvictUnit := gcPolicy.MaxContainers / evictUnits.NumEvictUnits()
		if numContainersPerEvictUnit < 1 {
			numContainersPerEvictUnit = 1
		}
		cgc.enforceMaxContainersPerEvictUnit(evictUnits, numContainersPerEvictUnit)

		// If we still need to evict, evict oldest first.
		numContainers := evictUnits.NumContainers()
		if numContainers > gcPolicy.MaxContainers {
			flattened := make([]containerGCInfo, 0, numContainers)
			for key := range evictUnits {
				flattened = append(flattened, evictUnits[key]...)
			}
			sort.Sort(byCreated(flattened))

			cgc.removeOldestN(flattened, numContainers-gcPolicy.MaxContainers)
		}
	}

	// Remove sandboxes with zero containers
	evictSandboxes, err := cgc.evictableSandboxes()
	if err != nil {
		return err
	}
	for _, sandbox := range evictSandboxes {
		cgc.removeSandbox(sandbox)
	}

	return nil
}
