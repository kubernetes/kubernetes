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

// Package reconciler implements interfaces that attempt to reconcile the
// desired state of the world with the actual state of the world by triggering
// relevant actions (attach, detach, mount, unmount).
package reconciler

import (
	"context"
	"fmt"
	"io/fs"
	"os"
	"path"
	"path/filepath"
	"time"

	"k8s.io/apimachinery/pkg/api/resource"

	"k8s.io/klog/v2"
	"k8s.io/mount-utils"
	utilpath "k8s.io/utils/path"
	utilstrings "k8s.io/utils/strings"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/pkg/kubelet/config"
	"k8s.io/kubernetes/pkg/kubelet/volumemanager/cache"
	"k8s.io/kubernetes/pkg/util/goroutinemap/exponentialbackoff"
	volumepkg "k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/util"
	"k8s.io/kubernetes/pkg/volume/util/hostutil"
	"k8s.io/kubernetes/pkg/volume/util/nestedpendingoperations"
	"k8s.io/kubernetes/pkg/volume/util/operationexecutor"
	volumetypes "k8s.io/kubernetes/pkg/volume/util/types"
)

// Reconciler runs a periodic loop to reconcile the desired state of the world
// with the actual state of the world by triggering attach, detach, mount, and
// unmount operations.
// Note: This is distinct from the Reconciler implemented by the attach/detach
// controller. This reconciles state for the kubelet volume manager. That
// reconciles state for the attach/detach controller.
type Reconciler interface {
	// Starts running the reconciliation loop which executes periodically, checks
	// if volumes that should be mounted are mounted and volumes that should
	// be unmounted are unmounted. If not, it will trigger mount/unmount
	// operations to rectify.
	// If attach/detach management is enabled, the manager will also check if
	// volumes that should be attached are attached and volumes that should
	// be detached are detached and trigger attach/detach operations as needed.
	Run(stopCh <-chan struct{})

	// StatesHasBeenSynced returns true only after syncStates process starts to sync
	// states at least once after kubelet starts
	StatesHasBeenSynced() bool
}

// NewReconciler returns a new instance of Reconciler.
//
// controllerAttachDetachEnabled - if true, indicates that the attach/detach
//   controller is responsible for managing the attach/detach operations for
//   this node, and therefore the volume manager should not
// loopSleepDuration - the amount of time the reconciler loop sleeps between
//   successive executions
// waitForAttachTimeout - the amount of time the Mount function will wait for
//   the volume to be attached
// nodeName - the Name for this node, used by Attach and Detach methods
// desiredStateOfWorld - cache containing the desired state of the world
// actualStateOfWorld - cache containing the actual state of the world
// populatorHasAddedPods - checker for whether the populator has finished
//   adding pods to the desiredStateOfWorld cache at least once after sources
//   are all ready (before sources are ready, pods are probably missing)
// operationExecutor - used to trigger attach/detach/mount/unmount operations
//   safely (prevents more than one operation from being triggered on the same
//   volume)
// mounter - mounter passed in from kubelet, passed down unmount path
// hostutil - hostutil passed in from kubelet
// volumePluginMgr - volume plugin manager passed from kubelet
func NewReconciler(
	kubeClient clientset.Interface,
	controllerAttachDetachEnabled bool,
	loopSleepDuration time.Duration,
	waitForAttachTimeout time.Duration,
	nodeName types.NodeName,
	desiredStateOfWorld cache.DesiredStateOfWorld,
	actualStateOfWorld cache.ActualStateOfWorld,
	populatorHasAddedPods func() bool,
	operationExecutor operationexecutor.OperationExecutor,
	mounter mount.Interface,
	hostutil hostutil.HostUtils,
	volumePluginMgr *volumepkg.VolumePluginMgr,
	kubeletPodsDir string) Reconciler {
	return &reconciler{
		kubeClient:                    kubeClient,
		controllerAttachDetachEnabled: controllerAttachDetachEnabled,
		loopSleepDuration:             loopSleepDuration,
		waitForAttachTimeout:          waitForAttachTimeout,
		nodeName:                      nodeName,
		desiredStateOfWorld:           desiredStateOfWorld,
		actualStateOfWorld:            actualStateOfWorld,
		populatorHasAddedPods:         populatorHasAddedPods,
		operationExecutor:             operationExecutor,
		mounter:                       mounter,
		hostutil:                      hostutil,
		volumePluginMgr:               volumePluginMgr,
		kubeletPodsDir:                kubeletPodsDir,
		timeOfLastSync:                time.Time{},
	}
}

type reconciler struct {
	kubeClient                    clientset.Interface
	controllerAttachDetachEnabled bool
	loopSleepDuration             time.Duration
	waitForAttachTimeout          time.Duration
	nodeName                      types.NodeName
	desiredStateOfWorld           cache.DesiredStateOfWorld
	actualStateOfWorld            cache.ActualStateOfWorld
	populatorHasAddedPods         func() bool
	operationExecutor             operationexecutor.OperationExecutor
	mounter                       mount.Interface
	hostutil                      hostutil.HostUtils
	volumePluginMgr               *volumepkg.VolumePluginMgr
	kubeletPodsDir                string
	timeOfLastSync                time.Time
}

func (rc *reconciler) Run(stopCh <-chan struct{}) {
	wait.Until(rc.reconciliationLoopFunc(), rc.loopSleepDuration, stopCh)
}

func (rc *reconciler) reconciliationLoopFunc() func() {
	return func() {
		rc.reconcile()

		// Sync the state with the reality once after all existing pods are added to the desired state from all sources.
		// Otherwise, the reconstruct process may clean up pods' volumes that are still in use because
		// desired state of world does not contain a complete list of pods.
		if rc.populatorHasAddedPods() && !rc.StatesHasBeenSynced() {
			klog.InfoS("Reconciler: start to sync state")
			rc.sync()
		}
	}
}

func (rc *reconciler) reconcile() {
	// Unmounts are triggered before mounts so that a volume that was
	// referenced by a pod that was deleted and is now referenced by another
	// pod is unmounted from the first pod before being mounted to the new
	// pod.
	rc.unmountVolumes()

	// Next we mount required volumes. This function could also trigger
	// attach if kubelet is responsible for attaching volumes.
	// If underlying PVC was resized while in-use then this function also handles volume
	// resizing.
	rc.mountOrAttachVolumes()

	// Ensure devices that should be detached/unmounted are detached/unmounted.
	rc.unmountDetachDevices()
}

func (rc *reconciler) unmountVolumes() {
	// Ensure volumes that should be unmounted are unmounted.
	for _, mountedVolume := range rc.actualStateOfWorld.GetAllMountedVolumes() {
		if !rc.desiredStateOfWorld.PodExistsInVolume(mountedVolume.PodName, mountedVolume.VolumeName) {
			// Volume is mounted, unmount it
			klog.V(5).InfoS(mountedVolume.GenerateMsgDetailed("Starting operationExecutor.UnmountVolume", ""))
			err := rc.operationExecutor.UnmountVolume(
				mountedVolume.MountedVolume, rc.actualStateOfWorld, rc.kubeletPodsDir)
			if err != nil && !isExpectedError(err) {
				klog.ErrorS(err, mountedVolume.GenerateErrorDetailed(fmt.Sprintf("operationExecutor.UnmountVolume failed (controllerAttachDetachEnabled %v)", rc.controllerAttachDetachEnabled), err).Error())
			}
			if err == nil {
				klog.InfoS(mountedVolume.GenerateMsgDetailed("operationExecutor.UnmountVolume started", ""))
			}
		}
	}
}

func (rc *reconciler) mountOrAttachVolumes() {
	// Ensure volumes that should be attached/mounted are attached/mounted.
	for _, volumeToMount := range rc.desiredStateOfWorld.GetVolumesToMount() {
		volMounted, devicePath, err := rc.actualStateOfWorld.PodExistsInVolume(volumeToMount.PodName, volumeToMount.VolumeName, volumeToMount.PersistentVolumeSize)
		volumeToMount.DevicePath = devicePath
		if cache.IsVolumeNotAttachedError(err) {
			rc.waitForVolumeAttach(volumeToMount)
		} else if !volMounted || cache.IsRemountRequiredError(err) {
			rc.mountAttachedVolumes(volumeToMount, err)
		} else if cache.IsFSResizeRequiredError(err) {
			fsResizeRequiredErr, _ := err.(cache.FsResizeRequiredError)
			rc.expandVolume(volumeToMount, fsResizeRequiredErr.CurrentSize)
		}
	}
}

func (rc *reconciler) expandVolume(volumeToMount cache.VolumeToMount, currentSize resource.Quantity) {
	klog.V(4).InfoS(volumeToMount.GenerateMsgDetailed("Starting operationExecutor.ExpandInUseVolume", ""), "pod", klog.KObj(volumeToMount.Pod))
	err := rc.operationExecutor.ExpandInUseVolume(volumeToMount.VolumeToMount, rc.actualStateOfWorld, currentSize)

	if err != nil && !isExpectedError(err) {
		klog.ErrorS(err, volumeToMount.GenerateErrorDetailed("operationExecutor.ExpandInUseVolume failed", err).Error(), "pod", klog.KObj(volumeToMount.Pod))
	}

	if err == nil {
		klog.V(4).InfoS(volumeToMount.GenerateMsgDetailed("operationExecutor.ExpandInUseVolume started", ""), "pod", klog.KObj(volumeToMount.Pod))
	}
}

func (rc *reconciler) mountAttachedVolumes(volumeToMount cache.VolumeToMount, podExistError error) {
	// Volume is not mounted, or is already mounted, but requires remounting
	remountingLogStr := ""
	isRemount := cache.IsRemountRequiredError(podExistError)
	if isRemount {
		remountingLogStr = "Volume is already mounted to pod, but remount was requested."
	}
	klog.V(4).InfoS(volumeToMount.GenerateMsgDetailed("Starting operationExecutor.MountVolume", remountingLogStr), "pod", klog.KObj(volumeToMount.Pod))
	err := rc.operationExecutor.MountVolume(
		rc.waitForAttachTimeout,
		volumeToMount.VolumeToMount,
		rc.actualStateOfWorld,
		isRemount)
	if err != nil && !isExpectedError(err) {
		klog.ErrorS(err, volumeToMount.GenerateErrorDetailed(fmt.Sprintf("operationExecutor.MountVolume failed (controllerAttachDetachEnabled %v)", rc.controllerAttachDetachEnabled), err).Error(), "pod", klog.KObj(volumeToMount.Pod))
	}
	if err == nil {
		if remountingLogStr == "" {
			klog.V(1).InfoS(volumeToMount.GenerateMsgDetailed("operationExecutor.MountVolume started", remountingLogStr), "pod", klog.KObj(volumeToMount.Pod))
		} else {
			klog.V(5).InfoS(volumeToMount.GenerateMsgDetailed("operationExecutor.MountVolume started", remountingLogStr), "pod", klog.KObj(volumeToMount.Pod))
		}
	}
}

func (rc *reconciler) waitForVolumeAttach(volumeToMount cache.VolumeToMount) {
	if rc.controllerAttachDetachEnabled || !volumeToMount.PluginIsAttachable {
		//// lets not spin a goroutine and unnecessarily trigger exponential backoff if this happens
		if volumeToMount.PluginIsAttachable && !volumeToMount.ReportedInUse {
			klog.V(5).InfoS(volumeToMount.GenerateMsgDetailed("operationExecutor.VerifyControllerAttachedVolume failed", " volume not marked in-use"), "pod", klog.KObj(volumeToMount.Pod))
			return
		}
		// Volume is not attached (or doesn't implement attacher), kubelet attach is disabled, wait
		// for controller to finish attaching volume.
		klog.V(5).InfoS(volumeToMount.GenerateMsgDetailed("Starting operationExecutor.VerifyControllerAttachedVolume", ""), "pod", klog.KObj(volumeToMount.Pod))
		err := rc.operationExecutor.VerifyControllerAttachedVolume(
			volumeToMount.VolumeToMount,
			rc.nodeName,
			rc.actualStateOfWorld)
		if err != nil && !isExpectedError(err) {
			klog.ErrorS(err, volumeToMount.GenerateErrorDetailed(fmt.Sprintf("operationExecutor.VerifyControllerAttachedVolume failed (controllerAttachDetachEnabled %v)", rc.controllerAttachDetachEnabled), err).Error(), "pod", klog.KObj(volumeToMount.Pod))
		}
		if err == nil {
			klog.InfoS(volumeToMount.GenerateMsgDetailed("operationExecutor.VerifyControllerAttachedVolume started", ""), "pod", klog.KObj(volumeToMount.Pod))
		}
	} else {
		// Volume is not attached to node, kubelet attach is enabled, volume implements an attacher,
		// so attach it
		volumeToAttach := operationexecutor.VolumeToAttach{
			VolumeName: volumeToMount.VolumeName,
			VolumeSpec: volumeToMount.VolumeSpec,
			NodeName:   rc.nodeName,
		}
		klog.V(5).InfoS(volumeToAttach.GenerateMsgDetailed("Starting operationExecutor.AttachVolume", ""), "pod", klog.KObj(volumeToMount.Pod))
		err := rc.operationExecutor.AttachVolume(volumeToAttach, rc.actualStateOfWorld)
		if err != nil && !isExpectedError(err) {
			klog.ErrorS(err, volumeToMount.GenerateErrorDetailed(fmt.Sprintf("operationExecutor.AttachVolume failed (controllerAttachDetachEnabled %v)", rc.controllerAttachDetachEnabled), err).Error(), "pod", klog.KObj(volumeToMount.Pod))
		}
		if err == nil {
			klog.InfoS(volumeToMount.GenerateMsgDetailed("operationExecutor.AttachVolume started", ""), "pod", klog.KObj(volumeToMount.Pod))
		}
	}
}

func (rc *reconciler) unmountDetachDevices() {
	for _, attachedVolume := range rc.actualStateOfWorld.GetUnmountedVolumes() {
		// Check IsOperationPending to avoid marking a volume as detached if it's in the process of mounting.
		if !rc.desiredStateOfWorld.VolumeExists(attachedVolume.VolumeName) &&
			!rc.operationExecutor.IsOperationPending(attachedVolume.VolumeName, nestedpendingoperations.EmptyUniquePodName, nestedpendingoperations.EmptyNodeName) {
			if attachedVolume.DeviceMayBeMounted() {
				// Volume is globally mounted to device, unmount it
				klog.V(5).InfoS(attachedVolume.GenerateMsgDetailed("Starting operationExecutor.UnmountDevice", ""))
				err := rc.operationExecutor.UnmountDevice(
					attachedVolume.AttachedVolume, rc.actualStateOfWorld, rc.hostutil)
				if err != nil && !isExpectedError(err) {
					klog.ErrorS(err, attachedVolume.GenerateErrorDetailed(fmt.Sprintf("operationExecutor.UnmountDevice failed (controllerAttachDetachEnabled %v)", rc.controllerAttachDetachEnabled), err).Error())
				}
				if err == nil {
					klog.InfoS(attachedVolume.GenerateMsgDetailed("operationExecutor.UnmountDevice started", ""))
				}
			} else {
				// Volume is attached to node, detach it
				// Kubelet not responsible for detaching or this volume has a non-attachable volume plugin.
				if rc.controllerAttachDetachEnabled || !attachedVolume.PluginIsAttachable {
					rc.actualStateOfWorld.MarkVolumeAsDetached(attachedVolume.VolumeName, attachedVolume.NodeName)
					klog.InfoS(attachedVolume.GenerateMsgDetailed("Volume detached", fmt.Sprintf("DevicePath %q", attachedVolume.DevicePath)))
				} else {
					// Only detach if kubelet detach is enabled
					klog.V(5).InfoS(attachedVolume.GenerateMsgDetailed("Starting operationExecutor.DetachVolume", ""))
					err := rc.operationExecutor.DetachVolume(
						attachedVolume.AttachedVolume, false /* verifySafeToDetach */, rc.actualStateOfWorld)
					if err != nil && !isExpectedError(err) {
						klog.ErrorS(err, attachedVolume.GenerateErrorDetailed(fmt.Sprintf("operationExecutor.DetachVolume failed (controllerAttachDetachEnabled %v)", rc.controllerAttachDetachEnabled), err).Error())
					}
					if err == nil {
						klog.InfoS(attachedVolume.GenerateMsgDetailed("operationExecutor.DetachVolume started", ""))
					}
				}
			}
		}
	}
}

// sync process tries to observe the real world by scanning all pods' volume directories from the disk.
// If the actual and desired state of worlds are not consistent with the observed world, it means that some
// mounted volumes are left out probably during kubelet restart. This process will reconstruct
// the volumes and update the actual and desired states. For the volumes that cannot support reconstruction,
// it will try to clean up the mount paths with operation executor.
func (rc *reconciler) sync() {
	defer rc.updateLastSyncTime()
	rc.syncStates()
}

func (rc *reconciler) updateLastSyncTime() {
	rc.timeOfLastSync = time.Now()
}

func (rc *reconciler) StatesHasBeenSynced() bool {
	return !rc.timeOfLastSync.IsZero()
}

type podVolume struct {
	podName        volumetypes.UniquePodName
	volumeSpecName string
	volumePath     string
	pluginName     string
	volumeMode     v1.PersistentVolumeMode
}

type reconstructedVolume struct {
	volumeName          v1.UniqueVolumeName
	podName             volumetypes.UniquePodName
	volumeSpec          *volumepkg.Spec
	outerVolumeSpecName string
	pod                 *v1.Pod
	volumeGidValue      string
	devicePath          string
	mounter             volumepkg.Mounter
	deviceMounter       volumepkg.DeviceMounter
	blockVolumeMapper   volumepkg.BlockVolumeMapper
}

// syncStates scans the volume directories under the given pod directory.
// If the volume is not in desired state of world, this function will reconstruct
// the volume related information and put it in both the actual and desired state of worlds.
// For some volume plugins that cannot support reconstruction, it will clean up the existing
// mount points since the volume is no long needed (removed from desired state)
func (rc *reconciler) syncStates() {
	// Get volumes information by reading the pod's directory
	podVolumes, err := getVolumesFromPodDir(rc.kubeletPodsDir)
	if err != nil {
		klog.ErrorS(err, "Cannot get volumes from disk, skip sync states for volume reconstruction")
		return
	}
	volumesNeedUpdate := make(map[v1.UniqueVolumeName]*reconstructedVolume)
	volumeNeedReport := []v1.UniqueVolumeName{}
	for _, volume := range podVolumes {
		if rc.actualStateOfWorld.VolumeExistsWithSpecName(volume.podName, volume.volumeSpecName) {
			klog.V(4).InfoS("Volume exists in actual state, skip cleaning up mounts", "podName", volume.podName, "volumeSpecName", volume.volumeSpecName)
			// There is nothing to reconstruct
			continue
		}
		volumeInDSW := rc.desiredStateOfWorld.VolumeExistsWithSpecName(volume.podName, volume.volumeSpecName)

		reconstructedVolume, err := rc.reconstructVolume(volume)
		if err != nil {
			if volumeInDSW {
				// Some pod needs the volume, don't clean it up and hope that
				// reconcile() calls SetUp and reconstructs the volume in ASW.
				klog.V(4).InfoS("Volume exists in desired state, skip cleaning up mounts", "podName", volume.podName, "volumeSpecName", volume.volumeSpecName)
				continue
			}
			// No pod needs the volume.
			klog.InfoS("Could not construct volume information, cleaning up mounts", "podName", volume.podName, "volumeSpecName", volume.volumeSpecName, "err", err)
			rc.cleanupMounts(volume)
			continue
		}
		if volumeInDSW {
			// Some pod needs the volume. And it exists on disk. Some previous
			// kubelet must have created the directory, therefore it must have
			// reported the volume as in use. Mark the volume as in use also in
			// this new kubelet so reconcile() calls SetUp and re-mounts the
			// volume if it's necessary.
			volumeNeedReport = append(volumeNeedReport, reconstructedVolume.volumeName)
			klog.V(4).InfoS("Volume exists in desired state, marking as InUse", "podName", volume.podName, "volumeSpecName", volume.volumeSpecName)
			continue
		}
		// There is no pod that uses the volume.
		if rc.operationExecutor.IsOperationPending(reconstructedVolume.volumeName, nestedpendingoperations.EmptyUniquePodName, nestedpendingoperations.EmptyNodeName) {
			klog.InfoS("Volume is in pending operation, skip cleaning up mounts")
		}
		klog.V(2).InfoS("Reconciler sync states: could not find pod information in desired state, update it in actual state", "reconstructedVolume", reconstructedVolume)
		volumesNeedUpdate[reconstructedVolume.volumeName] = reconstructedVolume
	}

	if len(volumesNeedUpdate) > 0 {
		if err = rc.updateStates(volumesNeedUpdate); err != nil {
			klog.ErrorS(err, "Error occurred during reconstruct volume from disk")
		}
	}
	if len(volumeNeedReport) > 0 {
		rc.desiredStateOfWorld.MarkVolumesReportedInUse(volumeNeedReport)
	}
}

func (rc *reconciler) cleanupMounts(volume podVolume) {
	klog.V(2).InfoS("Reconciler sync states: could not find volume information in desired state, clean up the mount points", "podName", volume.podName, "volumeSpecName", volume.volumeSpecName)
	mountedVolume := operationexecutor.MountedVolume{
		PodName:             volume.podName,
		VolumeName:          v1.UniqueVolumeName(volume.volumeSpecName),
		InnerVolumeSpecName: volume.volumeSpecName,
		PluginName:          volume.pluginName,
		PodUID:              types.UID(volume.podName),
	}
	// TODO: Currently cleanupMounts only includes UnmountVolume operation. In the next PR, we will add
	// to unmount both volume and device in the same routine.
	err := rc.operationExecutor.UnmountVolume(mountedVolume, rc.actualStateOfWorld, rc.kubeletPodsDir)
	if err != nil {
		klog.ErrorS(err, mountedVolume.GenerateErrorDetailed("volumeHandler.UnmountVolumeHandler for UnmountVolume failed", err).Error())
		return
	}
}

// Reconstruct volume data structure by reading the pod's volume directories
func (rc *reconciler) reconstructVolume(volume podVolume) (*reconstructedVolume, error) {
	// plugin initializations
	plugin, err := rc.volumePluginMgr.FindPluginByName(volume.pluginName)
	if err != nil {
		return nil, err
	}

	// Create pod object
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID: types.UID(volume.podName),
		},
	}
	mapperPlugin, err := rc.volumePluginMgr.FindMapperPluginByName(volume.pluginName)
	if err != nil {
		return nil, err
	}
	if volume.volumeMode == v1.PersistentVolumeBlock && mapperPlugin == nil {
		return nil, fmt.Errorf("could not find block volume plugin %q (spec.Name: %q) pod %q (UID: %q)", volume.pluginName, volume.volumeSpecName, volume.podName, pod.UID)
	}

	volumeSpec, err := rc.operationExecutor.ReconstructVolumeOperation(
		volume.volumeMode,
		plugin,
		mapperPlugin,
		pod.UID,
		volume.podName,
		volume.volumeSpecName,
		volume.volumePath,
		volume.pluginName)
	if err != nil {
		return nil, err
	}

	// We have to find the plugins by volume spec (NOT by plugin name) here
	// in order to correctly reconstruct ephemeral volume types.
	// Searching by spec checks whether the volume is actually attachable
	// (i.e. has a PV) whereas searching by plugin name can only tell whether
	// the plugin supports attachable volumes.
	attachablePlugin, err := rc.volumePluginMgr.FindAttachablePluginBySpec(volumeSpec)
	if err != nil {
		return nil, err
	}
	deviceMountablePlugin, err := rc.volumePluginMgr.FindDeviceMountablePluginBySpec(volumeSpec)
	if err != nil {
		return nil, err
	}

	var uniqueVolumeName v1.UniqueVolumeName
	if attachablePlugin != nil || deviceMountablePlugin != nil {
		uniqueVolumeName, err = util.GetUniqueVolumeNameFromSpec(plugin, volumeSpec)
		if err != nil {
			return nil, err
		}
	} else {
		uniqueVolumeName = util.GetUniqueVolumeNameFromSpecWithPod(volume.podName, plugin, volumeSpec)
	}

	var volumeMapper volumepkg.BlockVolumeMapper
	var volumeMounter volumepkg.Mounter
	var deviceMounter volumepkg.DeviceMounter
	// Path to the mount or block device to check
	var checkPath string

	if volume.volumeMode == v1.PersistentVolumeBlock {
		var newMapperErr error
		volumeMapper, newMapperErr = mapperPlugin.NewBlockVolumeMapper(
			volumeSpec,
			pod,
			volumepkg.VolumeOptions{})
		if newMapperErr != nil {
			return nil, fmt.Errorf(
				"reconstructVolume.NewBlockVolumeMapper failed for volume %q (spec.Name: %q) pod %q (UID: %q) with: %v",
				uniqueVolumeName,
				volumeSpec.Name(),
				volume.podName,
				pod.UID,
				newMapperErr)
		}
		mapDir, linkName := volumeMapper.GetPodDeviceMapPath()
		checkPath = filepath.Join(mapDir, linkName)
	} else {
		var err error
		volumeMounter, err = plugin.NewMounter(
			volumeSpec,
			pod,
			volumepkg.VolumeOptions{})
		if err != nil {
			return nil, fmt.Errorf(
				"reconstructVolume.NewMounter failed for volume %q (spec.Name: %q) pod %q (UID: %q) with: %v",
				uniqueVolumeName,
				volumeSpec.Name(),
				volume.podName,
				pod.UID,
				err)
		}
		checkPath = volumeMounter.GetPath()
		if deviceMountablePlugin != nil {
			deviceMounter, err = deviceMountablePlugin.NewDeviceMounter()
			if err != nil {
				return nil, fmt.Errorf("reconstructVolume.NewDeviceMounter failed for volume %q (spec.Name: %q) pod %q (UID: %q) with: %v",
					uniqueVolumeName,
					volumeSpec.Name(),
					volume.podName,
					pod.UID,
					err)
			}
		}
	}

	// Check existence of mount point for filesystem volume or symbolic link for block volume
	isExist, checkErr := rc.operationExecutor.CheckVolumeExistenceOperation(volumeSpec, checkPath, volumeSpec.Name(), rc.mounter, uniqueVolumeName, volume.podName, pod.UID, attachablePlugin)
	if checkErr != nil {
		return nil, checkErr
	}
	// If mount or symlink doesn't exist, volume reconstruction should be failed
	if !isExist {
		return nil, fmt.Errorf("volume: %q is not mounted", uniqueVolumeName)
	}

	reconstructedVolume := &reconstructedVolume{
		volumeName: uniqueVolumeName,
		podName:    volume.podName,
		volumeSpec: volumeSpec,
		// volume.volumeSpecName is actually InnerVolumeSpecName. It will not be used
		// for volume cleanup.
		// in case pod is added back to desired state, outerVolumeSpecName will be updated from dsw information.
		// See issue #103143 and its fix for details.
		outerVolumeSpecName: volume.volumeSpecName,
		pod:                 pod,
		deviceMounter:       deviceMounter,
		volumeGidValue:      "",
		// devicePath is updated during updateStates() by checking node status's VolumesAttached data.
		// TODO: get device path directly from the volume mount path.
		devicePath:        "",
		mounter:           volumeMounter,
		blockVolumeMapper: volumeMapper,
	}
	return reconstructedVolume, nil
}

// updateDevicePath gets the node status to retrieve volume device path information.
func (rc *reconciler) updateDevicePath(volumesNeedUpdate map[v1.UniqueVolumeName]*reconstructedVolume) {
	node, fetchErr := rc.kubeClient.CoreV1().Nodes().Get(context.TODO(), string(rc.nodeName), metav1.GetOptions{})
	if fetchErr != nil {
		klog.ErrorS(fetchErr, "UpdateStates in reconciler: could not get node status with error")
	} else {
		for _, attachedVolume := range node.Status.VolumesAttached {
			if volume, exists := volumesNeedUpdate[attachedVolume.Name]; exists {
				volume.devicePath = attachedVolume.DevicePath
				volumesNeedUpdate[attachedVolume.Name] = volume
				klog.V(4).InfoS("Update devicePath from node status for volume", "volumeName", attachedVolume.Name, "path", volume.devicePath)
			}
		}
	}
}

// getDeviceMountPath returns device mount path for block volume which
// implements BlockVolumeMapper or filesystem volume which implements
// DeviceMounter
func getDeviceMountPath(volume *reconstructedVolume) (string, error) {
	if volume.blockVolumeMapper != nil {
		// for block volume, we return its global map path
		return volume.blockVolumeMapper.GetGlobalMapPath(volume.volumeSpec)
	} else if volume.deviceMounter != nil {
		// for filesystem volume, we return its device mount path if the plugin implements DeviceMounter
		return volume.deviceMounter.GetDeviceMountPath(volume.volumeSpec)
	} else {
		return "", fmt.Errorf("blockVolumeMapper or deviceMounter required")
	}
}

func (rc *reconciler) updateStates(volumesNeedUpdate map[v1.UniqueVolumeName]*reconstructedVolume) error {
	// Get the node status to retrieve volume device path information.
	// Skip reporting devicePath in node objects if kubeClient is nil.
	// In standalone mode, kubelet is not expected to mount any attachable volume types or secret, configmaps etc.
	if rc.kubeClient != nil {
		rc.updateDevicePath(volumesNeedUpdate)
	}

	for _, volume := range volumesNeedUpdate {
		err := rc.actualStateOfWorld.MarkVolumeAsAttached(
			//TODO: the devicePath might not be correct for some volume plugins: see issue #54108
			volume.volumeName, volume.volumeSpec, "" /* nodeName */, volume.devicePath)
		if err != nil {
			klog.ErrorS(err, "Could not add volume information to actual state of world", "pod", klog.KObj(volume.pod))
			continue
		}
		markVolumeOpts := operationexecutor.MarkVolumeOpts{
			PodName:             volume.podName,
			PodUID:              types.UID(volume.podName),
			VolumeName:          volume.volumeName,
			Mounter:             volume.mounter,
			BlockVolumeMapper:   volume.blockVolumeMapper,
			OuterVolumeSpecName: volume.outerVolumeSpecName,
			VolumeGidVolume:     volume.volumeGidValue,
			VolumeSpec:          volume.volumeSpec,
			VolumeMountState:    operationexecutor.VolumeMounted,
		}
		err = rc.actualStateOfWorld.MarkVolumeAsMounted(markVolumeOpts)
		if err != nil {
			klog.ErrorS(err, "Could not add pod to volume information to actual state of world", "pod", klog.KObj(volume.pod))
			continue
		}
		klog.V(4).InfoS("Volume is marked as mounted and added into the actual state", "pod", klog.KObj(volume.pod), "podName", volume.podName, "volumeName", volume.volumeName)
		// If the volume has device to mount, we mark its device as mounted.
		if volume.deviceMounter != nil || volume.blockVolumeMapper != nil {
			deviceMountPath, err := getDeviceMountPath(volume)
			if err != nil {
				klog.ErrorS(err, "Could not find device mount path for volume", "volumeName", volume.volumeName, "pod", klog.KObj(volume.pod))
				continue
			}
			err = rc.actualStateOfWorld.MarkDeviceAsMounted(volume.volumeName, volume.devicePath, deviceMountPath)
			if err != nil {
				klog.ErrorS(err, "Could not mark device is mounted to actual state of world", "pod", klog.KObj(volume.pod))
				continue
			}
			klog.V(4).InfoS("Volume is marked device as mounted and added into the actual state", "pod", klog.KObj(volume.pod), "podName", volume.podName, "volumeName", volume.volumeName)
		}
	}
	return nil
}

// getVolumesFromPodDir scans through the volumes directories under the given pod directory.
// It returns a list of pod volume information including pod's uid, volume's plugin name, mount path,
// and volume spec name.
func getVolumesFromPodDir(podDir string) ([]podVolume, error) {
	podsDirInfo, err := os.ReadDir(podDir)
	if err != nil {
		return nil, err
	}
	volumes := []podVolume{}
	for i := range podsDirInfo {
		if !podsDirInfo[i].IsDir() {
			continue
		}
		podName := podsDirInfo[i].Name()
		podDir := path.Join(podDir, podName)

		// Find filesystem volume information
		// ex. filesystem volume: /pods/{podUid}/volume/{escapeQualifiedPluginName}/{volumeName}
		volumesDirs := map[v1.PersistentVolumeMode]string{
			v1.PersistentVolumeFilesystem: path.Join(podDir, config.DefaultKubeletVolumesDirName),
		}
		// Find block volume information
		// ex. block volume: /pods/{podUid}/volumeDevices/{escapeQualifiedPluginName}/{volumeName}
		volumesDirs[v1.PersistentVolumeBlock] = path.Join(podDir, config.DefaultKubeletVolumeDevicesDirName)

		for volumeMode, volumesDir := range volumesDirs {
			var volumesDirInfo []fs.DirEntry
			if volumesDirInfo, err = os.ReadDir(volumesDir); err != nil {
				// Just skip the loop because given volumesDir doesn't exist depending on volumeMode
				continue
			}
			for _, volumeDir := range volumesDirInfo {
				pluginName := volumeDir.Name()
				volumePluginPath := path.Join(volumesDir, pluginName)
				volumePluginDirs, err := utilpath.ReadDirNoStat(volumePluginPath)
				if err != nil {
					klog.ErrorS(err, "Could not read volume plugin directory", "volumePluginPath", volumePluginPath)
					continue
				}
				unescapePluginName := utilstrings.UnescapeQualifiedName(pluginName)
				for _, volumeName := range volumePluginDirs {
					volumePath := path.Join(volumePluginPath, volumeName)
					klog.V(5).InfoS("Volume path from volume plugin directory", "podName", podName, "volumePath", volumePath)
					volumes = append(volumes, podVolume{
						podName:        volumetypes.UniquePodName(podName),
						volumeSpecName: volumeName,
						volumePath:     volumePath,
						pluginName:     unescapePluginName,
						volumeMode:     volumeMode,
					})
				}
			}
		}
	}
	klog.V(4).InfoS("Get volumes from pod directory", "path", podDir, "volumes", volumes)
	return volumes, nil
}

// ignore nestedpendingoperations.IsAlreadyExists and exponentialbackoff.IsExponentialBackoff errors, they are expected.
func isExpectedError(err error) bool {
	return nestedpendingoperations.IsAlreadyExists(err) || exponentialbackoff.IsExponentialBackoff(err) || operationexecutor.IsMountFailedPreconditionError(err)
}
