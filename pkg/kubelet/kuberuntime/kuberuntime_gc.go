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
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/util/workqueue"
	internalapi "k8s.io/cri-api/pkg/apis"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	"k8s.io/klog/v2"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/cribuffer"
	"k8s.io/kubernetes/pkg/kubelet/inits"
	"k8s.io/kubernetes/pkg/kubelet/pleg"
	"k8s.io/kubernetes/pkg/kubelet/podworks"
	"k8s.io/kubernetes/pkg/util/mmap"
	"k8s.io/kubernetes/pkg/util/observer"
)

const (
	containerGCPeriod = time.Minute
)

// containerGC is the manager of garbage collection.
type containerGC struct {
	client                 internalapi.RuntimeService
	manager                *kubeGenericRuntimeManager
	podStateProvider       podStateProvider
	queue                  workqueue.RateLimitingInterface
	gcPolicy               kubecontainer.GCPolicy
	gcPolicySet            bool
	allSourcesReady        bool
	evictTerminatedPods    bool
	needEvict              bool
	evictUnits             mmap.Mmaper
	sandboxesEvictByPod    mmap.Mmaper
	aliveContainers        mmap.Mmaper
	aliveSandboxs          mmap.Mmaper
	fixEvictContainerEvent observer.SubjectEvent
	fixEvictSandboxEvent   observer.SubjectEvent
	fixLogEvent            observer.SubjectEvent
	fixErrorEvent          observer.SubjectEvent
	fixDirectGCEvent       observer.SubjectEvent
}

type directGCEventChannel struct {
	ch chan struct{}
}

type directGCEvent struct {
	allSourcesReady     bool
	evictTerminatedPods bool
	channel             *directGCEventChannel
}

// NewContainerGC creates a new containerGC.
func newContainerGC(client internalapi.RuntimeService, podStateProvider podStateProvider, manager *kubeGenericRuntimeManager) *containerGC {
	gc := &containerGC{
		client:              client,
		manager:             manager,
		podStateProvider:    podStateProvider,
		queue:               workqueue.NewRateLimitingQueue(workqueue.DefaultControllerRateLimiter()),
		evictUnits:          mmap.NewMmap3(),
		sandboxesEvictByPod: mmap.NewMmap2(),
		aliveContainers:     mmap.NewMmap2(),
		aliveSandboxs:       mmap.NewMmap2(),
	}

	gc.fixEvictContainerEvent = observer.NewSubjectEvent(gc.handleEvictContainerEvent, nil)

	gc.fixEvictSandboxEvent = observer.NewSubjectEvent(gc.handleEvictSandboxEvent, nil)

	gc.fixLogEvent = observer.NewSubjectEvent(gc.HandleLogEvict, nil)

	gc.fixErrorEvent = observer.NewSubjectEvent(gc.HandleErrorEvict, nil)

	e := directGCEvent{
		channel: &directGCEventChannel{make(chan struct{}, 1)},
	}
	gc.fixDirectGCEvent = observer.NewSubjectEvent(gc.HandleDirectGCEvent, e)

	inits.SafeInitFuncs.Regist(gc.SafeInit)

	return gc
}

// containerGCInfo is the internal information kept for containers being considered for GC.
type containerGCInfo struct {
	pid types.UID
	// The ID of the container.
	id           string
	podSandboxId string
	// The name of the container.
	name string
	// Creation time for the container.
	createTime time.Time
	// If true, the container is in unknown state. Garbage collector should try
	// to stop containers before removal.
	state kubecontainer.State
}

// sandboxGCInfo is the internal information kept for sandboxes being considered for GC.
type sandboxGCInfo struct {
	// The ID of the sandbox.
	id string
	// Creation time for the sandbox.
	createTime time.Time
}

// Newest first.
type byCreated []containerGCInfo

func (a byCreated) Len() int           { return len(a) }
func (a byCreated) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a byCreated) Less(i, j int) bool { return a[i].createTime.After(a[j].createTime) }

// Newest first.
type sandboxByCreated []sandboxGCInfo

func (a sandboxByCreated) Len() int           { return len(a) }
func (a sandboxByCreated) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a sandboxByCreated) Less(i, j int) bool { return a[i].createTime.After(a[j].createTime) }

func (cgc *containerGC) SafeInit() {
	cribuffer.CriBuffer.Attach(pleg.CONTAINER_DIED, cgc.SubjectEventAddNotify, cgc.HandleContainerDie)
	cribuffer.CriBuffer.Attach(pleg.SANDBOX_DIED, cgc.SubjectEventAddNotify, cgc.HandleSandBoxDie)
	cribuffer.CriBuffer.Attach(pleg.CONTAINER_LIVE, cgc.SubjectEventAddNotify, cgc.HandleContainerLive)
	cribuffer.CriBuffer.Attach(pleg.SANDBOX_LIVE, cgc.SubjectEventAddNotify, cgc.HandleSandBoxLive)
	cribuffer.CriBuffer.Attach(pleg.CONTAINER_REMOVED, cgc.SubjectEventAddNotify, cgc.HandleContainerRemove)
	cribuffer.CriBuffer.Attach(pleg.SANDBOX_REMOVED, cgc.SubjectEventAddNotify, cgc.HandleSandBoxRemove)

	podworks.Observer.Attach(podworks.POD_REMOVED, cgc.SubjectEventAddNotify, cgc.handlePodRemovedNotify)
}

func (cgc *containerGC) SubjectEventAddNotify(se observer.SubjectEvent) {
	cgc.queue.Add(se)
}

func (cgc *containerGC) handlePodRemovedNotify(_ interface{}) error {
	cgc.queue.AddAfter(cgc.fixEvictContainerEvent, containerGCPeriod)
	return nil
}

func (cgc *containerGC) HandleContainerRemove(event interface{}) error {
	if e, ok := event.(pleg.PlegSubjectEvent); ok {
		cgc.aliveContainers.Delete(e.ID, e.Cid)

		containerInfo := containerGCInfo{
			pid:          e.ID,
			id:           e.Cid,
			name:         e.Cname,
			createTime:   e.CreateTime,
			state:        e.State,
			podSandboxId: e.PodSandboxId,
		}

		cgc.deleteEvictUnits(&containerInfo)

		cgc.tryEvictSandboxByContainer(&containerInfo)
		cgc.queue.AddAfter(cgc.fixLogEvent, containerGCPeriod)
	}

	return nil
}

func (cgc *containerGC) HandleContainerDie(event interface{}) error {
	if e, ok := event.(pleg.PlegSubjectEvent); ok {
		cgc.aliveContainers.Delete(e.ID, e.Cid)

		// Separate containers by evict units.
		containerInfo := containerGCInfo{
			pid:          e.ID,
			id:           e.Cid,
			name:         e.Cname,
			createTime:   e.CreateTime,
			state:        e.State,
			podSandboxId: e.PodSandboxId,
		}

		cgc.evictUnits.Insert(containerInfo, containerInfo.pid, containerInfo.name, containerInfo.id)

		cgc.queue.AddAfter(cgc.fixEvictContainerEvent, containerGCPeriod)
	}

	return nil
}

func (cgc *containerGC) deleteEvictUnits(containerInfo *containerGCInfo) error {
	//删除evict中的相同容器
	cgc.evictUnits.Delete(containerInfo.pid, containerInfo.name, containerInfo.id)
	return nil
}

func (cgc *containerGC) HandleContainerLive(event interface{}) error {
	if e, ok := event.(pleg.PlegSubjectEvent); ok {
		containerInfo := containerGCInfo{
			pid:          e.ID,
			id:           e.Cid,
			name:         e.Cname,
			createTime:   e.CreateTime,
			state:        e.State,
			podSandboxId: e.PodSandboxId,
		}

		cgc.aliveContainers.Insert(containerInfo, e.ID, e.Cid)

		cgc.deleteEvictUnits(&containerInfo)

	}

	return nil
}

func (cgc *containerGC) HandleSandBoxRemove(event interface{}) error {
	if e, ok := event.(pleg.PlegSubjectEvent); ok {
		cgc.aliveSandboxs.Delete(e.ID, e.Cid)

		cgc.deleteEvictSandbox(&e)
	}

	return nil
}

func (cgc *containerGC) HandleSandBoxDie(event interface{}) error {
	if e, ok := event.(pleg.PlegSubjectEvent); ok {
		cgc.aliveSandboxs.Delete(e.ID, e.Cid)

		sandboxInfo := sandboxGCInfo{
			id:         e.Cid,
			createTime: e.CreateTime,
		}
		podUID := types.UID(e.ID)
		cgc.sandboxesEvictByPod.Insert(sandboxInfo, podUID, sandboxInfo.id)

		cgc.queue.AddAfter(cgc.fixEvictSandboxEvent, containerGCPeriod)
	}

	return nil
}

func (cgc *containerGC) deleteEvictSandbox(e *pleg.PlegSubjectEvent) error {
	cgc.sandboxesEvictByPod.Delete(e.ID, e.Cid)
	return nil
}

func (cgc *containerGC) HandleSandBoxLive(event interface{}) error {
	if e, ok := event.(pleg.PlegSubjectEvent); ok {
		cgc.aliveSandboxs.Insert(struct{}{}, e.ID, e.Cid)

		cgc.deleteEvictSandbox(&e)
	}

	return nil
}

func (cgc *containerGC) handleEvictContainerEvent(_ interface{}) error {
	cgc.needEvict = false
	err := cgc.evictContainers()
	if err == nil && cgc.needEvict {
		cgc.queue.AddAfter(cgc.fixEvictContainerEvent, containerGCPeriod)
	}

	return err
}

func (cgc *containerGC) handleEvictSandboxEvent(_ interface{}) error {
	return cgc.evictSandboxes()
}

func (cgc *containerGC) HandleLogEvict(_ interface{}) error {
	if err := cgc.evictPodLogsDirectories(); err != nil {
		cgc.queue.AddAfter(cgc.fixLogEvent, containerGCPeriod)
	}

	return nil
}

func (cgc *containerGC) HandleErrorEvict(e interface{}) error {
	if err := cgc.handleEvictContainerEvent(e); err != nil {
		return err
	}

	if err := cgc.evictSandboxes(); err != nil {
		return err
	}

	if err := cgc.evictPodLogsDirectories(); err != nil {
		return err
	}

	return nil
}

func (cgc *containerGC) HandleDirectGCEvent(event interface{}) error {
	var err error
	if e, ok := event.(directGCEvent); ok {
		allSourcesReady := cgc.allSourcesReady
		evictTerminatedPods := cgc.evictTerminatedPods
		cgc.allSourcesReady = e.allSourcesReady
		cgc.evictTerminatedPods = e.evictTerminatedPods

		err = cgc.HandleErrorEvict(event)

		cgc.allSourcesReady = allSourcesReady
		cgc.evictTerminatedPods = evictTerminatedPods

		e.channel.ch <- struct{}{}
	}

	return err
}

// enforceMaxContainersPerEvictUnit enforces MaxPerPodContainer for each evictUnit.
func (cgc *containerGC) enforceMaxContainersPerEvictUnit(MaxContainers int) error {
	visitFunc := func(v interface{}, k ...interface{}) (mmap.LEVEL, error) {
		unit := []containerGCInfo{}
		v.(mmap.Mmaper).GetLeafValues(func(v interface{}) {
			unit = append(unit, v.(containerGCInfo))
		})

		toRemove := len(unit) - MaxContainers
		if toRemove > 0 {
			err := cgc.removeOldestN(unit, toRemove)
			if err != nil {
				return mmap.LEVEL_MAX, err
			}
		}

		return mmap.LEVEL_0, nil
	}

	return cgc.evictUnits.Iterate(mmap.LEVEL_2, visitFunc)
}

// removeOldestN removes the oldest toRemove containers and returns the resulting slice.
func (cgc *containerGC) removeOldestN(containers []containerGCInfo, toRemove int) error {
	// Remove from oldest to newest (last to first).
	numToKeep := len(containers) - toRemove
	if numToKeep > 0 {
		sort.Sort(byCreated(containers))
	}

	now := time.Now()
	newestGCTime := now.Add(-cgc.gcPolicy.MinAge)
	for i := len(containers) - 1; i >= numToKeep; i-- {
		container := containers[i]
		if newestGCTime.Before(container.createTime) && now.After(container.createTime) {
			cgc.needEvict = true
			if numToKeep > 0 {
				break
			} else {
				continue
			}
		}

		if container.state == kubecontainer.ContainerStateUnknown {
			// Containers in known state could be running, we should try
			// to stop it before removal.
			id := kubecontainer.ContainerID{
				Type: cgc.manager.runtimeName,
				ID:   container.id,
			}
			message := "Container is in unknown state, try killing it before removal"
			if err := cgc.manager.killContainer(nil, id, container.name, message, reasonUnknown, nil); err != nil {
				klog.ErrorS(err, "Failed to stop container", "containerID", container.id)
				cgc.needEvict = true
				continue
			}
		}
		if err := cgc.manager.removeContainer(container.id); err != nil {
			klog.ErrorS(err, "Failed to remove container", "containerID", container.id)
			if !strings.Contains(err.Error(), "No such container") {
				err := fmt.Errorf("Failed to remove container %q: %v", container.id, err)
				return err
			}
		}

		cgc.evictUnits.Delete(container.pid, container.name, container.id)
		cgc.queue.Add(cgc.fixLogEvent)
		cgc.tryEvictSandboxByContainer(&container)
	}

	// Assume we removed the containers so that we're not too aggressive.
	return nil
}

func (cgc *containerGC) tryEvictSandboxByContainer(container *containerGCInfo) {
	if cgc.sandboxesEvictByPod.Exist(container.pid) {
		cgc.queue.Add(cgc.fixEvictSandboxEvent)
	}
}

// removeOldestNSandboxes removes the oldest inactive toRemove sandboxes and
// returns the resulting slice.
func (cgc *containerGC) removeOldestNSandboxes(podUID types.UID, sandboxes []sandboxGCInfo, toRemove int) error {
	numToKeep := len(sandboxes) - toRemove
	if numToKeep > 0 {
		sort.Sort(sandboxByCreated(sandboxes))
	}

	// Remove from oldest to newest (last to first).
	for i := len(sandboxes) - 1; i >= numToKeep; i-- {
		sandbox := sandboxes[i]
		if cgc.SandboxShouldBeRemove(podUID, sandbox.id) {
			if err := cgc.removeSandbox(sandbox.id); err != nil {
				return err
			}
			cgc.sandboxesEvictByPod.Delete(podUID, sandbox.id)
		}
	}

	return nil
}

func (cgc *containerGC) SandboxShouldBeRemove(podUID types.UID, sandboxid string) bool {
	visitFunc := func(v interface{}, k ...interface{}) (mmap.LEVEL, error) {
		cinfo := v.(containerGCInfo)
		if cinfo.podSandboxId == sandboxid {
			return mmap.LEVEL_MAX, fmt.Errorf("%#v has sandboxid %v", cinfo, sandboxid)
		}

		return mmap.LEVEL_0, nil
	}

	if err := cgc.aliveContainers.Iterate(mmap.LEVEL_1, visitFunc, podUID); err != nil {
		return false
	}

	if err := cgc.evictUnits.Iterate(mmap.LEVEL_1, visitFunc, podUID); err != nil {
		return false
	}

	return true
}

// removeSandbox removes the sandbox by sandboxID.
func (cgc *containerGC) removeSandbox(sandboxID string) error {
	klog.V(4).InfoS("Removing sandbox", "sandboxID", sandboxID)
	// In normal cases, kubelet should've already called StopPodSandbox before
	// GC kicks in. To guard against the rare cases where this is not true, try
	// stopping the sandbox before removing it.
	if err := cgc.client.StopPodSandbox(sandboxID); err != nil {
		klog.ErrorS(err, "Failed to stop sandbox before removing", "sandboxID", sandboxID)
		if !strings.Contains(err.Error(), "No such container") {
			return fmt.Errorf("Failed to stop sandbox %q before removing: %v", sandboxID, err)
		}
	}
	if err := cgc.client.RemovePodSandbox(sandboxID); err != nil {
		klog.ErrorS(err, "Failed to remove sandbox", "sandboxID", sandboxID)
		if !strings.Contains(err.Error(), "No such container") {
			return fmt.Errorf("Failed to remove sandbox %q: %v", sandboxID, err)
		}
	}

	cgc.queue.AddAfter(cgc.fixLogEvent, containerGCPeriod)

	return nil
}

// evict all containers that are evictable
func (cgc *containerGC) evictContainers() error {
	// Remove deleted pod containers if all sources are ready.
	if cgc.allSourcesReady {
		visitFunc := func(v interface{}, k ...interface{}) (mmap.LEVEL, error) {
			pid := k[0].(types.UID)
			if cgc.podStateProvider.ShouldPodContentBeRemoved(pid) || (cgc.evictTerminatedPods && cgc.podStateProvider.ShouldPodRuntimeBeRemoved(pid)) {
				unit := []containerGCInfo{}
				v.(mmap.Mmaper).GetLeafValues(func(v interface{}) {
					unit = append(unit, v.(containerGCInfo))
				})
				err := cgc.removeOldestN(unit, len(unit))
				if err != nil {
					return mmap.LEVEL_MAX, err
				}
			}

			return mmap.LEVEL_0, nil
		}

		if err := cgc.evictUnits.Iterate(mmap.LEVEL_2, visitFunc); err != nil {
			return err
		}
	}

	// Enforce max containers per evict unit.
	if cgc.gcPolicy.MaxPerPodContainer >= 0 {
		if err := cgc.enforceMaxContainersPerEvictUnit(cgc.gcPolicy.MaxPerPodContainer); err != nil {
			return err
		}
	}

	// Enforce max total number of containers.
	evictUnitNum := cgc.evictUnits.GetLevelKeyNum(mmap.LEVEL_2)
	numContainers := cgc.evictUnits.Num()
	if cgc.gcPolicy.MaxContainers >= 0 && numContainers > cgc.gcPolicy.MaxContainers {
		// Leave an equal number of containers per evict unit (min: 1).
		numContainersPerEvictUnit := cgc.gcPolicy.MaxContainers / evictUnitNum
		if numContainersPerEvictUnit < 1 {
			numContainersPerEvictUnit = 1
		}
		if err := cgc.enforceMaxContainersPerEvictUnit(numContainersPerEvictUnit); err != nil {
			return err
		}

		// If we still need to evict, evict oldest first.
		numContainers = cgc.evictUnits.Num()
		if numContainers > cgc.gcPolicy.MaxContainers {
			flattened := []containerGCInfo{}
			cgc.evictUnits.GetLeafValues(func(v interface{}) {
				flattened = append(flattened, v.(containerGCInfo))
			})

			err := cgc.removeOldestN(flattened, numContainers-cgc.gcPolicy.MaxContainers)
			if err != nil {
				return err
			}
		}
	}
	return nil
}

// evictSandboxes remove all evictable sandboxes. An evictable sandbox must
// meet the following requirements:
//   1. not in ready state
//   2. contains no containers.
//   3. belong to a non-existent (i.e., already removed) pod, or is not the
//      most recently created sandbox for the pod.
func (cgc *containerGC) evictSandboxes() error {
	visitFunc := func(v interface{}, k ...interface{}) (mmap.LEVEL, error) {
		podUID := k[0].(types.UID)

		sandboxes := []sandboxGCInfo{}
		v.(mmap.Mmaper).GetLeafValues(func(v interface{}) {
			sandboxes = append(sandboxes, v.(sandboxGCInfo))
		})

		toRemove := len(sandboxes)

		// Keep latest one if the pod still exists.
		if !cgc.aliveSandboxs.Exist(podUID) {
			toRemove = toRemove - 1
		}

		if cgc.podStateProvider.ShouldPodContentBeRemoved(podUID) || (cgc.evictTerminatedPods && cgc.podStateProvider.ShouldPodRuntimeBeRemoved(podUID)) {
			toRemove = len(sandboxes)
		}

		err := cgc.removeOldestNSandboxes(podUID, sandboxes, toRemove)
		if err != nil {
			return mmap.LEVEL_MAX, err
		}

		return mmap.LEVEL_0, nil
	}

	return cgc.sandboxesEvictByPod.Iterate(mmap.LEVEL_2, visitFunc)
}

// evictPodLogsDirectories evicts all evictable pod logs directories. Pod logs directories
// are evictable if there are no corresponding pods.
func (cgc *containerGC) evictPodLogsDirectories() error {
	osInterface := cgc.manager.osInterface
	if cgc.allSourcesReady {
		// Only remove pod logs directories when all sources are ready.
		dirs, err := osInterface.ReadDir(podLogsRootDirectory)
		if err != nil {
			return fmt.Errorf("failed to read podLogsRootDirectory %q: %v", podLogsRootDirectory, err)
		}
		for _, dir := range dirs {
			name := dir.Name()
			podUID := parsePodUIDFromLogsDirectory(name)
			if !cgc.podStateProvider.ShouldPodContentBeRemoved(podUID) {
				continue
			}
			klog.V(4).InfoS("Removing pod logs", "podUID", podUID)
			err := osInterface.RemoveAll(filepath.Join(podLogsRootDirectory, name))
			if err != nil {
				klog.ErrorS(err, "Failed to remove pod logs directory", "path", name)
			}
		}
	}

	// Remove dead container log symlinks.
	// TODO(random-liu): Remove this after cluster logging supports CRI container log path.
	logSymlinks, _ := osInterface.Glob(filepath.Join(legacyContainerLogsDir, fmt.Sprintf("*.%s", legacyLogSuffix)))
	for _, logSymlink := range logSymlinks {
		if _, err := osInterface.Stat(logSymlink); os.IsNotExist(err) {
			if containerID, err := getContainerIDFromLegacyLogSymlink(logSymlink); err == nil {
				resp, err := cgc.manager.runtimeService.ContainerStatus(containerID, false)
				if err != nil {
					// TODO: we should handle container not found (i.e. container was deleted) case differently
					// once https://github.com/kubernetes/kubernetes/issues/63336 is resolved
					klog.InfoS("Error getting ContainerStatus for containerID", "containerID", containerID, "err", err)
				} else {
					status := resp.GetStatus()
					if status == nil {
						klog.V(4).InfoS("Container status is nil")
						continue
					}
					if status.State != runtimeapi.ContainerState_CONTAINER_EXITED {
						// Here is how container log rotation works (see containerLogManager#rotateLatestLog):
						//
						// 1. rename current log to rotated log file whose filename contains current timestamp (fmt.Sprintf("%s.%s", log, timestamp))
						// 2. reopen the container log
						// 3. if #2 fails, rename rotated log file back to container log
						//
						// There is small but indeterministic amount of time during which log file doesn't exist (between steps #1 and #2, between #1 and #3).
						// Hence the symlink may be deemed unhealthy during that period.
						// See https://github.com/kubernetes/kubernetes/issues/52172
						//
						// We only remove unhealthy symlink for dead containers
						klog.V(5).InfoS("Container is still running, not removing symlink", "containerID", containerID, "path", logSymlink)
						continue
					}
				}
			} else {
				klog.V(4).InfoS("Unable to obtain container ID", "err", err)
			}
			err := osInterface.Remove(logSymlink)
			if err != nil {
				klog.ErrorS(err, "Failed to remove container log dead symlink", "path", logSymlink)
			} else {
				klog.V(4).InfoS("Removed symlink", "path", logSymlink)
			}
		}
	}
	return nil
}

func (cgc *containerGC) DirectGarbageCollect(_ kubecontainer.GCPolicy, allSourcesReady bool, evictTerminatedPods bool) error {
	de := cgc.fixDirectGCEvent.Event.(directGCEvent)
	de.allSourcesReady = allSourcesReady
	de.evictTerminatedPods = evictTerminatedPods
	cgc.fixDirectGCEvent.Event = de
	cgc.queue.Add(cgc.fixDirectGCEvent)

	<-de.channel.ch
	return nil
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
func (cgc *containerGC) GarbageCollect(gcPolicy kubecontainer.GCPolicy, allSourcesReady bool, evictNonDeletedPods bool) error {
	if !cgc.gcPolicySet {
		cgc.gcPolicy = gcPolicy
		cgc.gcPolicySet = true
	}

	item, quit := cgc.queue.Get()
	if quit {
		return fmt.Errorf("GarbageCollect Queue quit")
	}

	defer cgc.queue.Done(item)

	cgc.allSourcesReady = allSourcesReady
	cgc.evictTerminatedPods = evictNonDeletedPods

	var err error

	if se, ok := item.(observer.SubjectEvent); ok {
		err = se.Handle()
	}

	if err != nil {
		cgc.queue.AddAfter(cgc.fixErrorEvent, containerGCPeriod)
	}

	return err
}
