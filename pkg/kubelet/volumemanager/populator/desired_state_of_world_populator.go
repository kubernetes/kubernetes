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

/*
Package populator implements interfaces that monitor and keep the states of the
caches in sync with the "ground truth".
*/
package populator

import (
	"sync"
	"time"

	"k8s.io/klog/v2"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/config"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/volumemanager/cache"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/csimigration"
	"k8s.io/kubernetes/pkg/volume/util"
	volumetypes "k8s.io/kubernetes/pkg/volume/util/types"
)

// DesiredStateOfWorldPopulator periodically loops through the list of active
// pods and ensures that each one exists in the desired state of the world cache
// if it has volumes. It also verifies that the pods in the desired state of the
// world cache still exist, if not, it removes them.
type DesiredStateOfWorldPopulator interface {
	Run(sourcesReady config.SourcesReady, stopCh <-chan struct{})

	// ReprocessPod sets value for the specified pod in processedPods
	// to false, forcing it to be reprocessed. This is required to enable
	// remounting volumes on pod updates (volumes like Downward API volumes
	// depend on this behavior to ensure volume content is updated).
	ReprocessPod(podName volumetypes.UniquePodName)

	// HasAddedPods returns whether the populator has looped through the list
	// of active pods and added them to the desired state of the world cache,
	// at a time after sources are all ready, at least once. It does not
	// return true before sources are all ready because before then, there is
	// a chance many or all pods are missing from the list of active pods and
	// so few to none will have been added.
	HasAddedPods() bool
}

// PodStateProvider can determine if a pod is going to be terminated.
type PodStateProvider interface {
	ShouldPodContainersBeTerminating(types.UID) bool
	ShouldPodRuntimeBeRemoved(types.UID) bool
}

// PodManager is the subset of methods the manager needs to observe the actual state of the kubelet.
// See pkg/k8s.io/kubernetes/pkg/kubelet/pod.Manager for method godoc.
type PodManager interface {
	GetPodByUID(types.UID) (*v1.Pod, bool)
	GetPods() []*v1.Pod
}

// NewDesiredStateOfWorldPopulator returns a new instance of
// DesiredStateOfWorldPopulator.
//
// kubeClient - used to fetch PV and PVC objects from the API server
// loopSleepDuration - the amount of time the populator loop sleeps between
// successive executions
//
// podManager - the kubelet podManager that is the source of truth for the pods
// that exist on this host
//
// desiredStateOfWorld - the cache to populate
func NewDesiredStateOfWorldPopulator(
	kubeClient clientset.Interface,
	loopSleepDuration time.Duration,
	podManager PodManager,
	podStateProvider PodStateProvider,
	desiredStateOfWorld cache.DesiredStateOfWorld,
	actualStateOfWorld cache.ActualStateOfWorld,
	kubeContainerRuntime kubecontainer.Runtime,
	keepTerminatedPodVolumes bool,
	csiMigratedPluginManager csimigration.PluginManager,
	intreeToCSITranslator csimigration.InTreeToCSITranslator,
	volumePluginMgr *volume.VolumePluginMgr) DesiredStateOfWorldPopulator {
	return &desiredStateOfWorldPopulator{
		kubeClient:          kubeClient,
		loopSleepDuration:   loopSleepDuration,
		podManager:          podManager,
		podStateProvider:    podStateProvider,
		desiredStateOfWorld: desiredStateOfWorld,
		actualStateOfWorld:  actualStateOfWorld,
		pods: processedPods{
			processedPods: make(map[volumetypes.UniquePodName]bool)},
		kubeContainerRuntime:     kubeContainerRuntime,
		keepTerminatedPodVolumes: keepTerminatedPodVolumes,
		hasAddedPods:             false,
		hasAddedPodsLock:         sync.RWMutex{},
		csiMigratedPluginManager: csiMigratedPluginManager,
		intreeToCSITranslator:    intreeToCSITranslator,
		volumePluginMgr:          volumePluginMgr,
	}
}

type desiredStateOfWorldPopulator struct {
	kubeClient               clientset.Interface
	loopSleepDuration        time.Duration
	podManager               PodManager
	podStateProvider         PodStateProvider
	desiredStateOfWorld      cache.DesiredStateOfWorld
	actualStateOfWorld       cache.ActualStateOfWorld
	pods                     processedPods
	kubeContainerRuntime     kubecontainer.Runtime
	keepTerminatedPodVolumes bool
	hasAddedPods             bool
	hasAddedPodsLock         sync.RWMutex
	csiMigratedPluginManager csimigration.PluginManager
	intreeToCSITranslator    csimigration.InTreeToCSITranslator
	volumePluginMgr          *volume.VolumePluginMgr
}

type processedPods struct {
	processedPods map[volumetypes.UniquePodName]bool
	sync.RWMutex
}

func (dswp *desiredStateOfWorldPopulator) Run(sourcesReady config.SourcesReady, stopCh <-chan struct{}) {
	// Wait for the completion of a loop that started after sources are all ready, then set hasAddedPods accordingly
	klog.InfoS("Desired state populator starts to run")
	wait.PollUntil(dswp.loopSleepDuration, func() (bool, error) {
		done := sourcesReady.AllReady()
		dswp.populatorLoop()
		return done, nil
	}, stopCh)
	dswp.hasAddedPodsLock.Lock()
	if !dswp.hasAddedPods {
		klog.InfoS("Finished populating initial desired state of world")
		dswp.hasAddedPods = true
	}
	dswp.hasAddedPodsLock.Unlock()
	wait.Until(dswp.populatorLoop, dswp.loopSleepDuration, stopCh)
}

func (dswp *desiredStateOfWorldPopulator) ReprocessPod(
	podName volumetypes.UniquePodName) {
	dswp.markPodProcessingFailed(podName)
}

func (dswp *desiredStateOfWorldPopulator) HasAddedPods() bool {
	dswp.hasAddedPodsLock.RLock()
	defer dswp.hasAddedPodsLock.RUnlock()
	return dswp.hasAddedPods
}

func (dswp *desiredStateOfWorldPopulator) populatorLoop() {
	dswp.findAndAddNewPods()
	dswp.findAndRemoveDeletedPods()
}

// Iterate through all pods and add to desired state of world if they don't
// exist but should
func (dswp *desiredStateOfWorldPopulator) findAndAddNewPods() {
	// Map unique pod name to outer volume name to MountedVolume.
	mountedVolumesForPod := make(map[volumetypes.UniquePodName]map[string]cache.MountedVolume)
	for _, mountedVolume := range dswp.actualStateOfWorld.GetMountedVolumes() {
		mountedVolumes, exist := mountedVolumesForPod[mountedVolume.PodName]
		if !exist {
			mountedVolumes = make(map[string]cache.MountedVolume)
			mountedVolumesForPod[mountedVolume.PodName] = mountedVolumes
		}
		mountedVolumes[mountedVolume.OuterVolumeSpecName] = mountedVolume
	}

	for _, pod := range dswp.podManager.GetPods() {
		// Keep consistency of adding pod during reconstruction
		if dswp.hasAddedPods && dswp.podStateProvider.ShouldPodContainersBeTerminating(pod.UID) {
			// Do not (re)add volumes for pods that can't also be starting containers
			continue
		}

		if !dswp.hasAddedPods && dswp.podStateProvider.ShouldPodRuntimeBeRemoved(pod.UID) {
			// When kubelet restarts, we need to add pods to dsw if there is a possibility
			// that the container may still be running
			continue
		}

		dswp.processPodVolumes(pod, mountedVolumesForPod)
	}
}

// Iterate through all pods in desired state of world, and remove if they no
// longer exist
func (dswp *desiredStateOfWorldPopulator) findAndRemoveDeletedPods() {
	podsFromCache := make(map[volumetypes.UniquePodName]struct{})
	for _, volumeToMount := range dswp.desiredStateOfWorld.GetVolumesToMount() {
		podsFromCache[volumetypes.UniquePodName(volumeToMount.Pod.UID)] = struct{}{}
		pod, podExists := dswp.podManager.GetPodByUID(volumeToMount.Pod.UID)
		if podExists {

			// check if the attachability has changed for this volume
			if volumeToMount.PluginIsAttachable {
				attachableVolumePlugin, err := dswp.volumePluginMgr.FindAttachablePluginBySpec(volumeToMount.VolumeSpec)
				// only this means the plugin is truly non-attachable
				if err == nil && attachableVolumePlugin == nil {
					// It is not possible right now for a CSI plugin to be both attachable and non-deviceMountable
					// So the uniqueVolumeName should remain the same after the attachability change
					dswp.desiredStateOfWorld.MarkVolumeAttachability(volumeToMount.VolumeName, false)
					klog.InfoS("Volume changes from attachable to non-attachable", "volumeName", volumeToMount.VolumeName)
					continue
				}
			}

			// Exclude known pods that we expect to be running
			if !dswp.podStateProvider.ShouldPodRuntimeBeRemoved(pod.UID) {
				continue
			}
			if dswp.keepTerminatedPodVolumes {
				continue
			}
		}

		// Once a pod has been deleted from kubelet pod manager, do not delete
		// it immediately from volume manager. Instead, check the kubelet
		// pod state provider to verify that all containers in the pod have been
		// terminated.
		if !dswp.podStateProvider.ShouldPodRuntimeBeRemoved(volumeToMount.Pod.UID) {
			klog.V(4).InfoS("Pod still has one or more containers in the non-exited state and will not be removed from desired state", "pod", klog.KObj(volumeToMount.Pod))
			continue
		}
		var volumeToMountSpecName string
		if volumeToMount.VolumeSpec != nil {
			volumeToMountSpecName = volumeToMount.VolumeSpec.Name()
		}
		removed := dswp.actualStateOfWorld.PodRemovedFromVolume(volumeToMount.PodName, volumeToMount.VolumeName)
		if removed && podExists {
			klog.V(4).InfoS("Actual state does not yet have volume mount information and pod still exists in pod manager, skip removing volume from desired state", "pod", klog.KObj(volumeToMount.Pod), "podUID", volumeToMount.Pod.UID, "volumeName", volumeToMountSpecName)
			continue
		}
		klog.V(4).InfoS("Removing volume from desired state", "pod", klog.KObj(volumeToMount.Pod), "podUID", volumeToMount.Pod.UID, "volumeName", volumeToMountSpecName)
		dswp.desiredStateOfWorld.DeletePodFromVolume(
			volumeToMount.PodName, volumeToMount.VolumeName)
		dswp.deleteProcessedPod(volumeToMount.PodName)
	}

	// Cleanup orphanded entries from processedPods
	dswp.pods.Lock()
	orphanedPods := make([]volumetypes.UniquePodName, 0, len(dswp.pods.processedPods))
	for k := range dswp.pods.processedPods {
		if _, ok := podsFromCache[k]; !ok {
			orphanedPods = append(orphanedPods, k)
		}
	}
	dswp.pods.Unlock()
	for _, orphanedPod := range orphanedPods {
		uid := types.UID(orphanedPod)
		_, podExists := dswp.podManager.GetPodByUID(uid)
		if !podExists && dswp.podStateProvider.ShouldPodRuntimeBeRemoved(uid) {
			dswp.deleteProcessedPod(orphanedPod)
		}
	}

	podsWithError := dswp.desiredStateOfWorld.GetPodsWithErrors()
	for _, podName := range podsWithError {
		if _, podExists := dswp.podManager.GetPodByUID(types.UID(podName)); !podExists {
			dswp.desiredStateOfWorld.PopPodErrors(podName)
		}
	}
}

// processPodVolumes processes the volumes in the given pod and adds them to the
// desired state of the world.
func (dswp *desiredStateOfWorldPopulator) processPodVolumes(
	pod *v1.Pod,
	mountedVolumesForPod map[volumetypes.UniquePodName]map[string]cache.MountedVolume) {
	if pod == nil {
		return
	}

	uniquePodName := util.GetUniquePodName(pod)
	if dswp.podPreviouslyProcessed(uniquePodName) {
		return
	}

	allVolumesAdded := true
	podVolumeInfo := util.NewPodVolumeInfos(
		pod,
		dswp.csiMigratedPluginManager,
		dswp.intreeToCSITranslator,
		dswp.volumePluginMgr,
		dswp.kubeClient)

	// Process volume spec for each volume defined in pod
	for _, podVolume := range pod.Spec.Volumes {
		if !podVolumeInfo.IsVolumeMounted(podVolume.Name) && !podVolumeInfo.IsVolumeMapped(podVolume.Name) {
			// Volume is not used in the pod, ignore it.
			klog.V(4).InfoS("Skipping unused volume", "pod", klog.KObj(pod), "volumeName", podVolume.Name)
			continue
		}

		volumeInfo, err := podVolumeInfo.GetVolumeInfo(&podVolume)
		if err != nil {
			klog.ErrorS(err, "Error processing volume", "pod", klog.KObj(pod), "volumeName", podVolume.Name)
			dswp.desiredStateOfWorld.AddErrorToPod(uniquePodName, err.Error())
			allVolumesAdded = false
			continue
		}

		// Add volume to desired state of world
		uniqueVolumeName, err := dswp.desiredStateOfWorld.AddPodToVolume(pod, volumeInfo)
		if err != nil {
			klog.ErrorS(err, "Failed to add volume to desiredStateOfWorld", "pod", klog.KObj(pod), "volumeName", podVolume.Name, "volumeSpecName", volumeInfo.Spec.Name())
			dswp.desiredStateOfWorld.AddErrorToPod(uniquePodName, err.Error())
			allVolumesAdded = false
		} else {
			klog.V(4).InfoS("Added volume to desired state", "pod", klog.KObj(pod), "volumeName", podVolume.Name, "volumeSpecName", volumeInfo.Spec.Name())
		}
		if !utilfeature.DefaultFeatureGate.Enabled(features.NewVolumeManagerReconstruction) {
			// sync reconstructed volume. This is necessary only when the old-style reconstruction is still used.
			// With reconstruct_new.go, AWS.MarkVolumeAsMounted will update the outer spec name of previously
			// uncertain volumes.
			dswp.actualStateOfWorld.SyncReconstructedVolume(uniqueVolumeName, uniquePodName, podVolume.Name)
		}

		dswp.checkVolumeFSResize(pod, podVolume, volumeInfo.PVC, volumeInfo.Spec, uniquePodName, mountedVolumesForPod)
	}

	// some of the volume additions may have failed, should not mark this pod as fully processed
	if allVolumesAdded {
		dswp.markPodProcessed(uniquePodName)
		// New pod has been synced. Re-mount all volumes that need it
		// (e.g. DownwardAPI)
		dswp.actualStateOfWorld.MarkRemountRequired(uniquePodName)
		// Remove any stored errors for the pod, everything went well in this processPodVolumes
		dswp.desiredStateOfWorld.PopPodErrors(uniquePodName)
	} else if dswp.podHasBeenSeenOnce(uniquePodName) {
		// For the Pod which has been processed at least once, even though some volumes
		// may not have been reprocessed successfully this round, we still mark it as processed to avoid
		// processing it at a very high frequency. The pod will be reprocessed when volume manager calls
		// ReprocessPod() which is triggered by SyncPod.
		dswp.markPodProcessed(uniquePodName)
	}

}

// checkVolumeFSResize records desired PVC size for a volume mounted by the pod.
// It is used for comparison with actual size(coming from pvc.Status.Capacity) and calling
// volume expansion on the node if needed.
func (dswp *desiredStateOfWorldPopulator) checkVolumeFSResize(
	pod *v1.Pod,
	podVolume v1.Volume,
	pvc *v1.PersistentVolumeClaim,
	volumeSpec *volume.Spec,
	uniquePodName volumetypes.UniquePodName,
	mountedVolumesForPod map[volumetypes.UniquePodName]map[string]cache.MountedVolume) {

	// if a volumeSpec does not have PV or has InlineVolumeSpecForCSIMigration set or pvc is nil
	// we can't resize the volume and hence resizing should be skipped.
	if volumeSpec.PersistentVolume == nil || volumeSpec.InlineVolumeSpecForCSIMigration || pvc == nil {
		// Only PVC supports resize operation.
		return
	}

	uniqueVolumeName, exist := getUniqueVolumeName(uniquePodName, podVolume.Name, mountedVolumesForPod)
	if !exist {
		// Volume not exist in ASW, we assume it hasn't been mounted yet. If it needs resize,
		// it will be handled as offline resize(if it indeed hasn't been mounted yet),
		// or online resize in subsequent loop(after we confirm it has been mounted).
		return
	}
	// volumeSpec.ReadOnly is the value that determines if volume could be formatted when being mounted.
	// This is the same flag that determines filesystem resizing behaviour for offline resizing and hence
	// we should use it here. This value comes from Pod.spec.volumes.persistentVolumeClaim.readOnly.
	if volumeSpec.ReadOnly {
		// This volume is used as read only by this pod, we don't perform resize for read only volumes.
		klog.V(5).InfoS("Skip file system resize check for the volume, as the volume is mounted as readonly", "pod", klog.KObj(pod), "volumeName", podVolume.Name)
		return
	}
	pvCap := volumeSpec.PersistentVolume.Spec.Capacity.Storage()
	pvcStatusCap := pvc.Status.Capacity.Storage()
	dswp.desiredStateOfWorld.UpdatePersistentVolumeSize(uniqueVolumeName, pvCap)

	// in case the actualStateOfWorld was rebuild after kubelet restart ensure that claimSize is set to accurate value
	dswp.actualStateOfWorld.InitializeClaimSize(klog.TODO(), uniqueVolumeName, pvcStatusCap)
}

func getUniqueVolumeName(
	podName volumetypes.UniquePodName,
	outerVolumeSpecName string,
	mountedVolumesForPod map[volumetypes.UniquePodName]map[string]cache.MountedVolume) (v1.UniqueVolumeName, bool) {
	mountedVolumes, exist := mountedVolumesForPod[podName]
	if !exist {
		return "", false
	}
	mountedVolume, exist := mountedVolumes[outerVolumeSpecName]
	if !exist {
		return "", false
	}
	return mountedVolume.VolumeName, true
}

// podPreviouslyProcessed returns true if the volumes for this pod have already
// been processed/reprocessed by the populator. Otherwise, the volumes for this pod need to
// be reprocessed.
func (dswp *desiredStateOfWorldPopulator) podPreviouslyProcessed(
	podName volumetypes.UniquePodName) bool {
	dswp.pods.RLock()
	defer dswp.pods.RUnlock()

	return dswp.pods.processedPods[podName]
}

// markPodProcessingFailed marks the specified pod from processedPods as false to indicate that it failed processing
func (dswp *desiredStateOfWorldPopulator) markPodProcessingFailed(
	podName volumetypes.UniquePodName) {
	dswp.pods.Lock()
	dswp.pods.processedPods[podName] = false
	dswp.pods.Unlock()
}

// podHasBeenSeenOnce returns true if the pod has been seen by the popoulator
// at least once.
func (dswp *desiredStateOfWorldPopulator) podHasBeenSeenOnce(
	podName volumetypes.UniquePodName) bool {
	dswp.pods.RLock()
	_, exist := dswp.pods.processedPods[podName]
	dswp.pods.RUnlock()
	return exist
}

// markPodProcessed records that the volumes for the specified pod have been
// processed by the populator
func (dswp *desiredStateOfWorldPopulator) markPodProcessed(
	podName volumetypes.UniquePodName) {
	dswp.pods.Lock()
	defer dswp.pods.Unlock()

	dswp.pods.processedPods[podName] = true
}

// deleteProcessedPod removes the specified pod from processedPods
func (dswp *desiredStateOfWorldPopulator) deleteProcessedPod(
	podName volumetypes.UniquePodName) {
	dswp.pods.Lock()
	defer dswp.pods.Unlock()

	delete(dswp.pods.processedPods, podName)
}
