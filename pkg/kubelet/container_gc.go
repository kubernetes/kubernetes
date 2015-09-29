/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"fmt"
	"os"
	"path"
	"path/filepath"
	"sort"
	"time"

	docker "github.com/fsouza/go-dockerclient"
	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/kubelet/dockertools"
	"k8s.io/kubernetes/pkg/types"
)

// Specified a policy for garbage collecting containers.
type ContainerGCPolicy struct {
	// Minimum age at which a container can be garbage collected, zero for no limit.
	MinAge time.Duration

	// Max number of dead containers any single pod (UID, container name) pair is
	// allowed to have, less than zero for no limit.
	MaxPerPodContainer int

	// Max number of total dead containers, less than zero for no limit.
	MaxContainers int
}

// Manages garbage collection of dead containers.
//
// Implementation is thread-compatible.
type containerGC interface {
	// Garbage collect containers.
	GarbageCollect() error
}

// TODO(vmarmol): Preferentially remove pod infra containers.
type realContainerGC struct {
	// Docker client to use.
	dockerClient dockertools.DockerInterface

	// Policy for garbage collection.
	policy ContainerGCPolicy

	// The path to the symlinked docker logs
	containerLogsDir string
}

// New containerGC instance with the specified policy.
func newContainerGC(dockerClient dockertools.DockerInterface, policy ContainerGCPolicy) (containerGC, error) {
	if policy.MinAge < 0 {
		return nil, fmt.Errorf("invalid minimum garbage collection age: %v", policy.MinAge)
	}

	return &realContainerGC{
		dockerClient:     dockerClient,
		policy:           policy,
		containerLogsDir: containerLogsDir,
	}, nil
}

// Internal information kept for containers being considered for GC.
type containerGCInfo struct {
	// Docker ID of the container.
	id string

	// Docker name of the container.
	name string

	// Creation time for the container.
	createTime time.Time

	// Full pod name, including namespace in the format `namespace_podName`.
	// This comes from dockertools.ParseDockerName(...)
	podNameWithNamespace string

	// Container name in pod
	containerName string
}

// Containers are considered for eviction as units of (UID, container name) pair.
type evictUnit struct {
	// UID of the pod.
	uid types.UID

	// Name of the container in the pod.
	name string
}

type containersByEvictUnit map[evictUnit][]containerGCInfo

// Returns the number of containers in this map.
func (cu containersByEvictUnit) NumContainers() int {
	num := 0
	for key := range cu {
		num += len(cu[key])
	}

	return num
}

// Returns the number of pod in this map.
func (cu containersByEvictUnit) NumEvictUnits() int {
	return len(cu)
}

// Newest first.
type byCreated []containerGCInfo

func (a byCreated) Len() int           { return len(a) }
func (a byCreated) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a byCreated) Less(i, j int) bool { return a[i].createTime.After(a[j].createTime) }

func (cgc *realContainerGC) GarbageCollect() error {
	// Separate containers by evict units.
	evictUnits, unidentifiedContainers, err := cgc.evictableContainers()
	if err != nil {
		return err
	}

	// Remove unidentified containers.
	for _, container := range unidentifiedContainers {
		glog.Infof("Removing unidentified dead container %q with ID %q", container.name, container.id)
		err = cgc.dockerClient.RemoveContainer(docker.RemoveContainerOptions{ID: container.id, RemoveVolumes: true})
		if err != nil {
			glog.Warningf("Failed to remove unidentified dead container %q: %v", container.name, err)
		}
	}

	// Enforce max containers per evict unit.
	if cgc.policy.MaxPerPodContainer >= 0 {
		cgc.enforceMaxContainersPerEvictUnit(evictUnits, cgc.policy.MaxPerPodContainer)
	}

	// Enforce max total number of containers.
	if cgc.policy.MaxContainers >= 0 && evictUnits.NumContainers() > cgc.policy.MaxContainers {
		// Leave an equal number of containers per evict unit (min: 1).
		numContainersPerEvictUnit := cgc.policy.MaxContainers / evictUnits.NumEvictUnits()
		if numContainersPerEvictUnit < 1 {
			numContainersPerEvictUnit = 1
		}
		cgc.enforceMaxContainersPerEvictUnit(evictUnits, numContainersPerEvictUnit)

		// If we still need to evict, evict oldest first.
		numContainers := evictUnits.NumContainers()
		if numContainers > cgc.policy.MaxContainers {
			flattened := make([]containerGCInfo, 0, numContainers)
			for uid := range evictUnits {
				flattened = append(flattened, evictUnits[uid]...)
			}
			sort.Sort(byCreated(flattened))

			cgc.removeOldestN(flattened, numContainers-cgc.policy.MaxContainers)
		}
	}

	// Remove dead symlinks - should only happen on upgrade
	// from a k8s version without proper log symlink cleanup
	logSymlinks, _ := filepath.Glob(path.Join(cgc.containerLogsDir, fmt.Sprintf("*.%s", dockertools.LogSuffix)))
	for _, logSymlink := range logSymlinks {
		if _, err = os.Stat(logSymlink); os.IsNotExist(err) {
			err = os.Remove(logSymlink)
			if err != nil {
				glog.Warningf("Failed to remove container log dead symlink %q: %v", logSymlink, err)
			}
		}
	}

	return nil
}

func (cgc *realContainerGC) enforceMaxContainersPerEvictUnit(evictUnits containersByEvictUnit, MaxContainers int) {
	for uid := range evictUnits {
		toRemove := len(evictUnits[uid]) - MaxContainers

		if toRemove > 0 {
			evictUnits[uid] = cgc.removeOldestN(evictUnits[uid], toRemove)
		}
	}
}

// Removes the oldest toRemove containers and returns the resulting slice.
func (cgc *realContainerGC) removeOldestN(containers []containerGCInfo, toRemove int) []containerGCInfo {
	// Remove from oldest to newest (last to first).
	numToKeep := len(containers) - toRemove
	for i := numToKeep; i < len(containers); i++ {
		err := cgc.dockerClient.RemoveContainer(docker.RemoveContainerOptions{ID: containers[i].id, RemoveVolumes: true})
		if err != nil {
			glog.Warningf("Failed to remove dead container %q: %v", containers[i].name, err)
		}
		symlinkPath := dockertools.LogSymlink(cgc.containerLogsDir, containers[i].podNameWithNamespace, containers[i].containerName, containers[i].id)
		err = os.Remove(symlinkPath)
		if err != nil && !os.IsNotExist(err) {
			glog.Warningf("Failed to remove container %q log symlink %q: %v", containers[i].name, symlinkPath, err)
		}
	}

	// Assume we removed the containers so that we're not too aggressive.
	return containers[:numToKeep]
}

// Get all containers that are evictable. Evictable containers are: not running
// and created more than MinAge ago.
func (cgc *realContainerGC) evictableContainers() (containersByEvictUnit, []containerGCInfo, error) {
	containers, err := dockertools.GetKubeletDockerContainers(cgc.dockerClient, true)
	if err != nil {
		return containersByEvictUnit{}, []containerGCInfo{}, err
	}

	unidentifiedContainers := make([]containerGCInfo, 0)
	evictUnits := make(containersByEvictUnit)
	newestGCTime := time.Now().Add(-cgc.policy.MinAge)
	for _, container := range containers {
		// Prune out running containers.
		data, err := cgc.dockerClient.InspectContainer(container.ID)
		if err != nil {
			// Container may have been removed already, skip.
			continue
		} else if data.State.Running {
			continue
		} else if newestGCTime.Before(data.Created) {
			continue
		}

		containerInfo := containerGCInfo{
			id:         container.ID,
			name:       container.Names[0],
			createTime: data.Created,
		}

		containerName, _, err := dockertools.ParseDockerName(container.Names[0])

		if err != nil {
			unidentifiedContainers = append(unidentifiedContainers, containerInfo)
		} else {
			key := evictUnit{
				uid:  containerName.PodUID,
				name: containerName.ContainerName,
			}
			containerInfo.podNameWithNamespace = containerName.PodFullName
			containerInfo.containerName = containerName.ContainerName
			evictUnits[key] = append(evictUnits[key], containerInfo)
		}
	}

	// Sort the containers by age.
	for uid := range evictUnits {
		sort.Sort(byCreated(evictUnits[uid]))
	}

	return evictUnits, unidentifiedContainers, nil
}
