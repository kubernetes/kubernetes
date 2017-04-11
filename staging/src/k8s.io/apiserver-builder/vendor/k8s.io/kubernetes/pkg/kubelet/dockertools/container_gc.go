/*
Copyright 2014 The Kubernetes Authors.

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

package dockertools

import (
	"fmt"
	"os"
	"path"
	"path/filepath"
	"sort"
	"time"

	dockertypes "github.com/docker/engine-api/types"
	"github.com/golang/glog"
	"k8s.io/apimachinery/pkg/types"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	knetwork "k8s.io/kubernetes/pkg/kubelet/network"
)

type containerGC struct {
	client           DockerInterface
	podGetter        podGetter
	network          *knetwork.PluginManager
	containerLogsDir string
}

func NewContainerGC(client DockerInterface, podGetter podGetter, network *knetwork.PluginManager, containerLogsDir string) *containerGC {
	return &containerGC{
		client:           client,
		podGetter:        podGetter,
		network:          network,
		containerLogsDir: containerLogsDir,
	}
}

// Internal information kept for containers being considered for GC.
type containerGCInfo struct {
	// Docker ID of the container.
	id string

	// Docker name of the container.
	dockerName string

	// Creation time for the container.
	createTime time.Time

	// Full pod name, including namespace in the format `namespace_podName`.
	// This comes from dockertools.ParseDockerName(...)
	podNameWithNamespace string

	// Kubernetes pod UID
	podUID types.UID

	// Container name in pod
	containerName string

	// Container network mode
	isHostNetwork bool
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

func (cgc *containerGC) enforceMaxContainersPerEvictUnit(evictUnits containersByEvictUnit, MaxContainers int) {
	for uid := range evictUnits {
		toRemove := len(evictUnits[uid]) - MaxContainers

		if toRemove > 0 {
			evictUnits[uid] = cgc.removeOldestN(evictUnits[uid], toRemove)
		}
	}
}

// Removes the oldest toRemove containers and returns the resulting slice.
func (cgc *containerGC) removeOldestN(containers []containerGCInfo, toRemove int) []containerGCInfo {
	// Remove from oldest to newest (last to first).
	numToKeep := len(containers) - toRemove
	for i := numToKeep; i < len(containers); i++ {
		cgc.removeContainer(containers[i])
	}

	// Assume we removed the containers so that we're not too aggressive.
	return containers[:numToKeep]
}

// Returns a full GC info structure on success, or a partial one on failure
func newContainerGCInfo(id string, inspectResult *dockertypes.ContainerJSON, created time.Time) (containerGCInfo, error) {
	containerName, _, err := ParseDockerName(inspectResult.Name)
	if err != nil {
		return containerGCInfo{
			id:         id,
			dockerName: inspectResult.Name,
		}, fmt.Errorf("failed to parse docker name %q: %v", inspectResult.Name, err)
	}

	networkMode := getDockerNetworkMode(inspectResult)
	return containerGCInfo{
		id:                   id,
		dockerName:           inspectResult.Name,
		podNameWithNamespace: containerName.PodFullName,
		podUID:               containerName.PodUID,
		containerName:        containerName.ContainerName,
		createTime:           created,
		isHostNetwork:        networkMode == namespaceModeHost,
	}, nil
}

// Get all containers that are evictable. Evictable containers are: not running
// and created more than MinAge ago.
func (cgc *containerGC) evictableContainers(minAge time.Duration) (containersByEvictUnit, []containerGCInfo, []containerGCInfo, error) {
	containers, err := GetKubeletDockerContainers(cgc.client, true)
	if err != nil {
		return containersByEvictUnit{}, []containerGCInfo{}, []containerGCInfo{}, err
	}

	unidentifiedContainers := make([]containerGCInfo, 0)
	netContainers := make([]containerGCInfo, 0)
	evictUnits := make(containersByEvictUnit)
	newestGCTime := time.Now().Add(-minAge)
	for _, container := range containers {
		// Prune out running containers.
		data, err := cgc.client.InspectContainer(container.ID)
		if err != nil {
			// Container may have been removed already, skip.
			continue
		} else if data.State.Running {
			continue
		}

		created, err := ParseDockerTimestamp(data.Created)
		if err != nil {
			glog.Errorf("Failed to parse Created timestamp %q for container %q", data.Created, container.ID)
		}
		if newestGCTime.Before(created) {
			continue
		}

		containerInfo, err := newContainerGCInfo(container.ID, data, created)
		if err != nil {
			unidentifiedContainers = append(unidentifiedContainers, containerInfo)
		} else {
			// Track net containers for special cleanup
			if containerIsNetworked(containerInfo.containerName) {
				netContainers = append(netContainers, containerInfo)
			}

			key := evictUnit{
				uid:  containerInfo.podUID,
				name: containerInfo.containerName,
			}
			evictUnits[key] = append(evictUnits[key], containerInfo)
		}
	}

	// Sort the containers by age.
	for uid := range evictUnits {
		sort.Sort(byCreated(evictUnits[uid]))
	}

	return evictUnits, netContainers, unidentifiedContainers, nil
}

// GarbageCollect removes dead containers using the specified container gc policy
func (cgc *containerGC) GarbageCollect(gcPolicy kubecontainer.ContainerGCPolicy, allSourcesReady bool) error {
	// Separate containers by evict units.
	evictUnits, netContainers, unidentifiedContainers, err := cgc.evictableContainers(gcPolicy.MinAge)
	if err != nil {
		return err
	}

	// Remove unidentified containers.
	for _, container := range unidentifiedContainers {
		glog.Infof("Removing unidentified dead container %q", container.dockerName)
		err = cgc.client.RemoveContainer(container.id, dockertypes.ContainerRemoveOptions{RemoveVolumes: true})
		if err != nil {
			glog.Warningf("Failed to remove unidentified dead container %q: %v", container.dockerName, err)
		}
	}

	// Always clean up net containers to ensure network resources are released
	// TODO: this may tear down networking again if the container doesn't get
	// removed in this GC cycle, but that already happens elsewhere...
	for _, container := range netContainers {
		glog.Infof("Cleaning up dead net container %q", container.dockerName)
		cgc.netContainerCleanup(container)
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
			for uid := range evictUnits {
				flattened = append(flattened, evictUnits[uid]...)
			}
			sort.Sort(byCreated(flattened))

			cgc.removeOldestN(flattened, numContainers-gcPolicy.MaxContainers)
		}
	}

	// Remove dead symlinks - should only happen on upgrade
	// from a k8s version without proper log symlink cleanup
	logSymlinks, _ := filepath.Glob(path.Join(cgc.containerLogsDir, fmt.Sprintf("*.%s", LogSuffix)))
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

func (cgc *containerGC) netContainerCleanup(containerInfo containerGCInfo) {
	if containerInfo.isHostNetwork {
		return
	}

	podName, podNamespace, err := kubecontainer.ParsePodFullName(containerInfo.podNameWithNamespace)
	if err != nil {
		glog.Warningf("failed to parse container %q pod full name: %v", containerInfo.dockerName, err)
		return
	}

	containerID := kubecontainer.DockerID(containerInfo.id).ContainerID()
	if err := cgc.network.TearDownPod(podNamespace, podName, containerID); err != nil {
		glog.Warningf("failed to tear down container %q network: %v", containerInfo.dockerName, err)
	}
}

func (cgc *containerGC) removeContainer(containerInfo containerGCInfo) {
	glog.V(4).Infof("Removing container %q", containerInfo.dockerName)
	err := cgc.client.RemoveContainer(containerInfo.id, dockertypes.ContainerRemoveOptions{RemoveVolumes: true})
	if err != nil {
		glog.Warningf("Failed to remove container %q: %v", containerInfo.dockerName, err)
	}
	symlinkPath := LogSymlink(cgc.containerLogsDir, containerInfo.podNameWithNamespace, containerInfo.containerName, containerInfo.id)
	err = os.Remove(symlinkPath)
	if err != nil && !os.IsNotExist(err) {
		glog.Warningf("Failed to remove container %q log symlink %q: %v", containerInfo.dockerName, symlinkPath, err)
	}
}

func (cgc *containerGC) deleteContainer(id string) error {
	data, err := cgc.client.InspectContainer(id)
	if err != nil {
		glog.Warningf("Failed to inspect container %q: %v", id, err)
		return err
	}
	if data.State.Running {
		return fmt.Errorf("container %q is still running", id)
	}

	containerInfo, err := newContainerGCInfo(id, data, time.Now())
	if err != nil {
		return err
	}

	if containerIsNetworked(containerInfo.containerName) {
		cgc.netContainerCleanup(containerInfo)
	}

	cgc.removeContainer(containerInfo)
	return nil
}

func (cgc *containerGC) isPodDeleted(podUID types.UID) bool {
	_, found := cgc.podGetter.GetPodByUID(podUID)
	return !found
}
