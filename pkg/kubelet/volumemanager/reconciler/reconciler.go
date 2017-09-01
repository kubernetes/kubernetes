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
// desired state of the with the actual state of the world by triggering
// relevant actions (attach, detach, mount, unmount).
package reconciler

import (
	"fmt"
	"io/ioutil"
	"path"
	"time"

	"github.com/golang/glog"
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/cmd/kubelet/app/options"
	"k8s.io/kubernetes/pkg/kubelet/volumemanager/cache"
	utilfile "k8s.io/kubernetes/pkg/util/file"
	"k8s.io/kubernetes/pkg/util/goroutinemap/exponentialbackoff"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/util/strings"
	volumepkg "k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/util/nestedpendingoperations"
	"k8s.io/kubernetes/pkg/volume/util/operationexecutor"
	volumetypes "k8s.io/kubernetes/pkg/volume/util/types"
	"k8s.io/kubernetes/pkg/volume/util/volumehelper"
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
//   syncDuration - the amount of time the syncStates sleeps between
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
// volumePluginMrg - volume plugin manager passed from kubelet
func NewReconciler(
	kubeClient clientset.Interface,
	controllerAttachDetachEnabled bool,
	loopSleepDuration time.Duration,
	syncDuration time.Duration,
	waitForAttachTimeout time.Duration,
	nodeName types.NodeName,
	desiredStateOfWorld cache.DesiredStateOfWorld,
	actualStateOfWorld cache.ActualStateOfWorld,
	populatorHasAddedPods func() bool,
	operationExecutor operationexecutor.OperationExecutor,
	mounter mount.Interface,
	volumePluginMgr *volumepkg.VolumePluginMgr,
	kubeletPodsDir string) Reconciler {
	return &reconciler{
		kubeClient:                    kubeClient,
		controllerAttachDetachEnabled: controllerAttachDetachEnabled,
		loopSleepDuration:             loopSleepDuration,
		syncDuration:                  syncDuration,
		waitForAttachTimeout:          waitForAttachTimeout,
		nodeName:                      nodeName,
		desiredStateOfWorld:           desiredStateOfWorld,
		actualStateOfWorld:            actualStateOfWorld,
		populatorHasAddedPods:         populatorHasAddedPods,
		operationExecutor:             operationExecutor,
		mounter:                       mounter,
		volumePluginMgr:               volumePluginMgr,
		kubeletPodsDir:                kubeletPodsDir,
		timeOfLastSync:                time.Time{},
	}
}

type reconciler struct {
	kubeClient                    clientset.Interface
	controllerAttachDetachEnabled bool
	loopSleepDuration             time.Duration
	syncDuration                  time.Duration
	waitForAttachTimeout          time.Duration
	nodeName                      types.NodeName
	desiredStateOfWorld           cache.DesiredStateOfWorld
	actualStateOfWorld            cache.ActualStateOfWorld
	populatorHasAddedPods         func() bool
	operationExecutor             operationexecutor.OperationExecutor
	mounter                       mount.Interface
	volumePluginMgr               *volumepkg.VolumePluginMgr
	kubeletPodsDir                string
	timeOfLastSync                time.Time
}

func (rc *reconciler) Run(stopCh <-chan struct{}) {
	// Wait for the populator to indicate that it has actually populated the desired state of world, meaning it has
	// completed a populate loop that started after sources are all ready. After, there's no need to keep checking.
	wait.PollUntil(rc.loopSleepDuration, func() (bool, error) {
		rc.reconciliationLoopFunc(rc.populatorHasAddedPods())()
		return rc.populatorHasAddedPods(), nil
	}, stopCh)
	wait.Until(rc.reconciliationLoopFunc(true), rc.loopSleepDuration, stopCh)
}

func (rc *reconciler) reconciliationLoopFunc(populatorHasAddedPods bool) func() {
	return func() {
		rc.reconcile()

		// Add a check that the populator has added pods so that reconciler's reconstruct process will start
		// after desired state of world is populated with pod volume information. Otherwise, reconciler's
		// reconstruct process may add incomplete volume information and cause confusion. In addition, if the
		// desired state of world has not been populated yet, the reconstruct process may clean up pods' volumes
		// that are still in use because desired state of world does not contain a complete list of pods.
		if populatorHasAddedPods && time.Since(rc.timeOfLastSync) > rc.syncDuration {
			glog.V(5).Infof("Desired state of world has been populated with pods, starting reconstruct state function")
			rc.sync()
		}
	}
}

func (rc *reconciler) reconcile() {
	// Unmounts are triggered before mounts so that a volume that was
	// referenced by a pod that was deleted and is now referenced by another
	// pod is unmounted from the first pod before being mounted to the new
	// pod.

	// Ensure volumes that should be unmounted are unmounted.
	for _, mountedVolume := range rc.actualStateOfWorld.GetMountedVolumes() {
		if !rc.desiredStateOfWorld.PodExistsInVolume(mountedVolume.PodName, mountedVolume.VolumeName) {
			// Volume is mounted, unmount it
			glog.V(12).Infof(mountedVolume.GenerateMsgDetailed("Starting operationExecutor.UnmountVolume", ""))
			err := rc.operationExecutor.UnmountVolume(
				mountedVolume.MountedVolume, rc.actualStateOfWorld)
			if err != nil &&
				!nestedpendingoperations.IsAlreadyExists(err) &&
				!exponentialbackoff.IsExponentialBackoff(err) {
				// Ignore nestedpendingoperations.IsAlreadyExists and exponentialbackoff.IsExponentialBackoff errors, they are expected.
				// Log all other errors.
				glog.Errorf(mountedVolume.GenerateErrorDetailed(fmt.Sprintf("operationExecutor.UnmountVolume failed (controllerAttachDetachEnabled %v)", rc.controllerAttachDetachEnabled), err).Error())
			}
			if err == nil {
				glog.Infof(mountedVolume.GenerateMsgDetailed("operationExecutor.UnmountVolume started", ""))
			}
		}
	}

	// Ensure volumes that should be attached/mounted are attached/mounted.
	for _, volumeToMount := range rc.desiredStateOfWorld.GetVolumesToMount() {
		volMounted, devicePath, err := rc.actualStateOfWorld.PodExistsInVolume(volumeToMount.PodName, volumeToMount.VolumeName)
		volumeToMount.DevicePath = devicePath
		if cache.IsVolumeNotAttachedError(err) {
			if rc.controllerAttachDetachEnabled || !volumeToMount.PluginIsAttachable {
				// Volume is not attached (or doesn't implement attacher), kubelet attach is disabled, wait
				// for controller to finish attaching volume.
				glog.V(12).Infof(volumeToMount.GenerateMsgDetailed("Starting operationExecutor.VerifyControllerAttachedVolume", ""))
				err := rc.operationExecutor.VerifyControllerAttachedVolume(
					volumeToMount.VolumeToMount,
					rc.nodeName,
					rc.actualStateOfWorld)
				if err != nil &&
					!nestedpendingoperations.IsAlreadyExists(err) &&
					!exponentialbackoff.IsExponentialBackoff(err) {
					// Ignore nestedpendingoperations.IsAlreadyExists and exponentialbackoff.IsExponentialBackoff errors, they are expected.
					// Log all other errors.
					glog.Errorf(volumeToMount.GenerateErrorDetailed(fmt.Sprintf("operationExecutor.VerifyControllerAttachedVolume failed (controllerAttachDetachEnabled %v)", rc.controllerAttachDetachEnabled), err).Error())
				}
				if err == nil {
					glog.Infof(volumeToMount.GenerateMsgDetailed("operationExecutor.VerifyControllerAttachedVolume started", ""))
				}
			} else {
				// Volume is not attached to node, kubelet attach is enabled, volume implements an attacher,
				// so attach it
				volumeToAttach := operationexecutor.VolumeToAttach{
					VolumeName: volumeToMount.VolumeName,
					VolumeSpec: volumeToMount.VolumeSpec,
					NodeName:   rc.nodeName,
				}
				glog.V(12).Infof(volumeToAttach.GenerateMsgDetailed("Starting operationExecutor.AttachVolume", ""))
				err := rc.operationExecutor.AttachVolume(volumeToAttach, rc.actualStateOfWorld)
				if err != nil &&
					!nestedpendingoperations.IsAlreadyExists(err) &&
					!exponentialbackoff.IsExponentialBackoff(err) {
					// Ignore nestedpendingoperations.IsAlreadyExists and exponentialbackoff.IsExponentialBackoff errors, they are expected.
					// Log all other errors.
					glog.Errorf(volumeToMount.GenerateErrorDetailed(fmt.Sprintf("operationExecutor.AttachVolume failed (controllerAttachDetachEnabled %v)", rc.controllerAttachDetachEnabled), err).Error())
				}
				if err == nil {
					glog.Infof(volumeToMount.GenerateMsgDetailed("operationExecutor.AttachVolume started", ""))
				}
			}
		} else if !volMounted || cache.IsRemountRequiredError(err) {
			// Volume is not mounted, or is already mounted, but requires remounting
			remountingLogStr := ""
			isRemount := cache.IsRemountRequiredError(err)
			if isRemount {
				remountingLogStr = "Volume is already mounted to pod, but remount was requested."
			}
			glog.V(12).Infof(volumeToMount.GenerateMsgDetailed("Starting operationExecutor.MountVolume", remountingLogStr))
			err := rc.operationExecutor.MountVolume(
				rc.waitForAttachTimeout,
				volumeToMount.VolumeToMount,
				rc.actualStateOfWorld,
				isRemount)
			if err != nil &&
				!nestedpendingoperations.IsAlreadyExists(err) &&
				!exponentialbackoff.IsExponentialBackoff(err) {
				// Ignore nestedpendingoperations.IsAlreadyExists and exponentialbackoff.IsExponentialBackoff errors, they are expected.
				// Log all other errors.
				glog.Errorf(volumeToMount.GenerateErrorDetailed(fmt.Sprintf("operationExecutor.MountVolume failed (controllerAttachDetachEnabled %v)", rc.controllerAttachDetachEnabled), err).Error())
			}
			if err == nil {
				if remountingLogStr == "" {
					glog.V(1).Infof(volumeToMount.GenerateMsgDetailed("operationExecutor.MountVolume started", remountingLogStr))
				} else {
					glog.V(5).Infof(volumeToMount.GenerateMsgDetailed("operationExecutor.MountVolume started", remountingLogStr))
				}
			}
		}
	}

	// Ensure devices that should be detached/unmounted are detached/unmounted.
	for _, attachedVolume := range rc.actualStateOfWorld.GetUnmountedVolumes() {
		// Check IsOperationPending to avoid marking a volume as detached if it's in the process of mounting.
		if !rc.desiredStateOfWorld.VolumeExists(attachedVolume.VolumeName) &&
			!rc.operationExecutor.IsOperationPending(attachedVolume.VolumeName, nestedpendingoperations.EmptyUniquePodName) {
			if attachedVolume.GloballyMounted {
				// Volume is globally mounted to device, unmount it
				glog.V(12).Infof(attachedVolume.GenerateMsgDetailed("Starting operationExecutor.UnmountDevice", ""))
				err := rc.operationExecutor.UnmountDevice(
					attachedVolume.AttachedVolume, rc.actualStateOfWorld, rc.mounter)
				if err != nil &&
					!nestedpendingoperations.IsAlreadyExists(err) &&
					!exponentialbackoff.IsExponentialBackoff(err) {
					// Ignore nestedpendingoperations.IsAlreadyExists and exponentialbackoff.IsExponentialBackoff errors, they are expected.
					// Log all other errors.
					glog.Errorf(attachedVolume.GenerateErrorDetailed(fmt.Sprintf("operationExecutor.UnmountDevice failed (controllerAttachDetachEnabled %v)", rc.controllerAttachDetachEnabled), err).Error())
				}
				if err == nil {
					glog.Infof(attachedVolume.GenerateMsgDetailed("operationExecutor.UnmountDevice started", ""))
				}
			} else {
				// Volume is attached to node, detach it
				// Kubelet not responsible for detaching or this volume has a non-attachable volume plugin.
				if rc.controllerAttachDetachEnabled || !attachedVolume.PluginIsAttachable {
					rc.actualStateOfWorld.MarkVolumeAsDetached(attachedVolume.VolumeName, attachedVolume.NodeName)
					glog.Infof(attachedVolume.GenerateMsgDetailed("Volume detached", fmt.Sprintf("DevicePath %q", attachedVolume.DevicePath)))
				} else {
					// Only detach if kubelet detach is enabled
					glog.V(12).Infof(attachedVolume.GenerateMsgDetailed("Starting operationExecutor.DetachVolume", ""))
					err := rc.operationExecutor.DetachVolume(
						attachedVolume.AttachedVolume, false /* verifySafeToDetach */, rc.actualStateOfWorld)
					if err != nil &&
						!nestedpendingoperations.IsAlreadyExists(err) &&
						!exponentialbackoff.IsExponentialBackoff(err) {
						// Ignore nestedpendingoperations.IsAlreadyExists && exponentialbackoff.IsExponentialBackoff errors, they are expected.
						// Log all other errors.
						glog.Errorf(attachedVolume.GenerateErrorDetailed(fmt.Sprintf("operationExecutor.DetachVolume failed (controllerAttachDetachEnabled %v)", rc.controllerAttachDetachEnabled), err).Error())
					}
					if err == nil {
						glog.Infof(attachedVolume.GenerateMsgDetailed("operationExecutor.DetachVolume started", ""))
					}
				}
			}
		}
	}
}

// sync process tries to observe the real world by scanning all pods' volume directories from the disk.
// If the actual and desired state of worlds are not consistent with the observed world, it means that some
// mounted volumes are left out probably during kubelet restart. This process will reconstruct
// the volumes and udpate the actual and desired states. In the following reconciler loop, those volumes will
// be cleaned up.
func (rc *reconciler) sync() {
	defer rc.updateLastSyncTime()
	rc.syncStates(rc.kubeletPodsDir)
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
	mountPath      string
	pluginName     string
}

type reconstructedVolume struct {
	volumeName          v1.UniqueVolumeName
	podName             volumetypes.UniquePodName
	volumeSpec          *volumepkg.Spec
	outerVolumeSpecName string
	pod                 *v1.Pod
	pluginIsAttachable  bool
	volumeGidValue      string
	devicePath          string
	reportedInUse       bool
	mounter             volumepkg.Mounter
}

// reconstructFromDisk scans the volume directories under the given pod directory. If the volume is not
// in either actual or desired state of world, or pending operation, this function will reconstruct
// the volume spec and put it in both the actual and desired state of worlds. If no running
// container is mounting the volume, the volume will be removed by desired state of world's populator and
// cleaned up by the reconciler.
func (rc *reconciler) syncStates(podsDir string) {
	// Get volumes information by reading the pod's directory
	podVolumes, err := getVolumesFromPodDir(podsDir)
	if err != nil {
		glog.Errorf("Cannot get volumes from disk %v", err)
		return
	}

	volumesNeedUpdate := make(map[v1.UniqueVolumeName]*reconstructedVolume)
	for _, volume := range podVolumes {
		reconstructedVolume, err := rc.reconstructVolume(volume)
		if err != nil {
			glog.Errorf("Could not construct volume information: %v", err)
			continue
		}
		// Check if there is an pending operation for the given pod and volume.
		// Need to check pending operation before checking the actual and desired
		// states to avoid race condition during checking. For example, the following
		// might happen if pending operation is checked after checking actual and desired states.
		// 1. Checking the pod and it does not exist in either actual or desired state.
		// 2. An operation for the given pod finishes and the actual state is updated.
		// 3. Checking and there is no pending operation for the given pod.
		// During state reconstruction period, no new volume operations could be issued. If the
		// mounted path is not in either pending operation, or actual or desired states, this
		// volume needs to be reconstructed back to the states.
		pending := rc.operationExecutor.IsOperationPending(reconstructedVolume.volumeName, reconstructedVolume.podName)
		dswExist := rc.desiredStateOfWorld.PodExistsInVolume(reconstructedVolume.podName, reconstructedVolume.volumeName)
		aswExist, _, _ := rc.actualStateOfWorld.PodExistsInVolume(reconstructedVolume.podName, reconstructedVolume.volumeName)

		if !rc.StatesHasBeenSynced() {
			// In case this is the first time to reconstruct state after kubelet starts, for a persistant volume, it must have
			// been mounted before kubelet restarts because no mount operations could be started at this time (node
			// status has not yet been updated before this very first syncStates finishes, so that VerifyControllerAttachedVolume will fail),
			// In this case, the volume state should be put back to actual state now no matter desired state has it or not.
			// This is to prevent node status from being updated to empty for attachable volumes. This might happen because
			// in the case that a volume is discovered on disk, and it is part of desired state, but is then quickly deleted
			// from the desired state. If in such situation, the volume is not added to the actual state, the node status updater will
			// not get this volume from either actual or desired state. In turn, this might cause master controller
			// detaching while the volume is still mounted.
			if aswExist || !reconstructedVolume.pluginIsAttachable {
				continue
			}
		} else {
			// Check pending first since no new operations could be started at this point.
			// Otherwise there might a race condition in checking actual states and pending operations
			if pending || dswExist || aswExist {
				continue
			}
		}

		glog.V(2).Infof(
			"Reconciler sync states: could not find pod information in desired or actual states or pending operation, update it in both states: %+v",
			reconstructedVolume)
		volumesNeedUpdate[reconstructedVolume.volumeName] = reconstructedVolume

	}

	if len(volumesNeedUpdate) > 0 {
		if err = rc.updateStates(volumesNeedUpdate); err != nil {
			glog.Errorf("Error occurred during reconstruct volume from disk: %v", err)
		}
	}

}

// Reconstruct Volume object and reconstructedVolume data structure by reading the pod's volume directories
func (rc *reconciler) reconstructVolume(volume podVolume) (*reconstructedVolume, error) {
	plugin, err := rc.volumePluginMgr.FindPluginByName(volume.pluginName)
	if err != nil {
		return nil, err
	}
	volumeSpec, err := plugin.ConstructVolumeSpec(volume.volumeSpecName, volume.mountPath)
	if err != nil {
		return nil, err
	}
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID: types.UID(volume.podName),
		},
	}
	attachablePlugin, err := rc.volumePluginMgr.FindAttachablePluginByName(volume.pluginName)
	if err != nil {
		return nil, err
	}

	volumeName, err := plugin.GetVolumeName(volumeSpec)
	if err != nil {
		return nil, err
	}
	var uniqueVolumeName v1.UniqueVolumeName
	if attachablePlugin != nil {
		uniqueVolumeName = volumehelper.GetUniqueVolumeName(volume.pluginName, volumeName)
	} else {
		uniqueVolumeName = volumehelper.GetUniqueVolumeNameForNonAttachableVolume(volume.podName, plugin, volumeSpec)
	}

	volumeMounter, newMounterErr := plugin.NewMounter(
		volumeSpec,
		pod,
		volumepkg.VolumeOptions{})
	if newMounterErr != nil {
		return nil, fmt.Errorf(
			"MountVolume.NewMounter failed for volume %q (spec.Name: %q) pod %q (UID: %q) with: %v",
			uniqueVolumeName,
			volumeSpec.Name(),
			volume.podName,
			pod.UID,
			newMounterErr)
	}

	reconstructedVolume := &reconstructedVolume{
		volumeName: uniqueVolumeName,
		podName:    volume.podName,
		volumeSpec: volumeSpec,
		// volume.volumeSpecName is actually InnerVolumeSpecName. But this information will likely to be updated in updateStates()
		// by checking the desired state volumeToMount list and getting the real OuterVolumeSpecName.
		// In case the pod is deleted during this period and desired state does not have this information, it will not be used
		// for volume cleanup.
		outerVolumeSpecName: volume.volumeSpecName,
		pod:                 pod,
		pluginIsAttachable:  attachablePlugin != nil,
		volumeGidValue:      "",
		devicePath:          "",
		mounter:             volumeMounter,
	}
	return reconstructedVolume, nil
}

func (rc *reconciler) updateStates(volumesNeedUpdate map[v1.UniqueVolumeName]*reconstructedVolume) error {
	// Get the node status to retrieve volume device path information.
	node, fetchErr := rc.kubeClient.Core().Nodes().Get(string(rc.nodeName), metav1.GetOptions{})
	if fetchErr != nil {
		glog.Errorf("updateStates in reconciler: could not get node status with error %v", fetchErr)
	} else {
		for _, attachedVolume := range node.Status.VolumesAttached {
			if volume, exists := volumesNeedUpdate[attachedVolume.Name]; exists {
				volume.devicePath = attachedVolume.DevicePath
				volumesNeedUpdate[attachedVolume.Name] = volume
				glog.V(4).Infof("Update devicePath from node status for volume (%q): %q", attachedVolume.Name, volume.devicePath)
			}
		}
	}

	// Get the list of volumes from desired state and update OuterVolumeSpecName if the information is available
	volumesToMount := rc.desiredStateOfWorld.GetVolumesToMount()
	for _, volumeToMount := range volumesToMount {
		if volume, exists := volumesNeedUpdate[volumeToMount.VolumeName]; exists {
			volume.outerVolumeSpecName = volumeToMount.OuterVolumeSpecName
			volumesNeedUpdate[volumeToMount.VolumeName] = volume
			glog.V(4).Infof("Update OuterVolumeSpecName from desired state for volume (%q): %q",
				volumeToMount.VolumeName, volume.outerVolumeSpecName)
		}
	}

	for _, volume := range volumesNeedUpdate {
		err := rc.actualStateOfWorld.MarkVolumeAsAttached(
			volume.volumeName, volume.volumeSpec, "" /* nodeName */, volume.devicePath)
		if err != nil {
			glog.Errorf("Could not add volume information to actual state of world: %v", err)
			continue
		}

		err = rc.actualStateOfWorld.AddPodToVolume(
			volume.podName,
			types.UID(volume.podName),
			volume.volumeName,
			volume.mounter,
			volume.outerVolumeSpecName,
			volume.volumeGidValue)
		if err != nil {
			glog.Errorf("Could not add pod to volume information to actual state of world: %v", err)
			continue
		}
		if volume.pluginIsAttachable {
			err = rc.actualStateOfWorld.MarkDeviceAsMounted(volume.volumeName)
			if err != nil {
				glog.Errorf("Could not mark device is mounted to actual state of world: %v", err)
				continue
			}
		}
		_, err = rc.desiredStateOfWorld.AddPodToVolume(volume.podName,
			volume.pod,
			volume.volumeSpec,
			volume.outerVolumeSpecName,
			volume.volumeGidValue)
		if err != nil {
			glog.Errorf("Could not add pod to volume information to desired state of world: %v", err)
		}
	}
	return nil
}

// getVolumesFromPodDir scans through the volumes directories under the given pod directory.
// It returns a list of pod volume information including pod's uid, volume's plugin name, mount path,
// and volume spec name.
func getVolumesFromPodDir(podDir string) ([]podVolume, error) {
	podsDirInfo, err := ioutil.ReadDir(podDir)
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
		volumesDir := path.Join(podDir, options.DefaultKubeletVolumesDirName)
		volumesDirInfo, err := ioutil.ReadDir(volumesDir)
		if err != nil {
			glog.Errorf("Could not read volume directory %q: %v", volumesDir, err)
			continue
		}
		for _, volumeDir := range volumesDirInfo {
			pluginName := volumeDir.Name()
			volumePluginPath := path.Join(volumesDir, pluginName)

			volumePluginDirs, err := utilfile.ReadDirNoStat(volumePluginPath)
			if err != nil {
				glog.Errorf("Could not read volume plugin directory %q: %v", volumePluginPath, err)
				continue
			}

			unescapePluginName := strings.UnescapeQualifiedNameForDisk(pluginName)
			for _, volumeName := range volumePluginDirs {
				mountPath := path.Join(volumePluginPath, volumeName)
				volumes = append(volumes, podVolume{
					podName:        volumetypes.UniquePodName(podName),
					volumeSpecName: volumeName,
					mountPath:      mountPath,
					pluginName:     unescapePluginName,
				})
			}

		}
	}
	glog.V(10).Infof("Get volumes from pod directory %q %+v", podDir, volumes)
	return volumes, nil
}
