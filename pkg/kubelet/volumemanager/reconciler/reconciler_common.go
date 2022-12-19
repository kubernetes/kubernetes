/*
Copyright 2022 The Kubernetes Authors.

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

package reconciler

import (
	"fmt"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/types"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/volumemanager/cache"
	"k8s.io/kubernetes/pkg/util/goroutinemap/exponentialbackoff"
	volumepkg "k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/util/hostutil"
	"k8s.io/kubernetes/pkg/volume/util/nestedpendingoperations"
	"k8s.io/kubernetes/pkg/volume/util/operationexecutor"
	"k8s.io/mount-utils"
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
// controller is responsible for managing the attach/detach operations for
// this node, and therefore the volume manager should not
//
// loopSleepDuration - the amount of time the reconciler loop sleeps between
// successive executions
//
// waitForAttachTimeout - the amount of time the Mount function will wait for
// the volume to be attached
//
// nodeName - the Name for this node, used by Attach and Detach methods
//
// desiredStateOfWorld - cache containing the desired state of the world
//
// actualStateOfWorld - cache containing the actual state of the world
//
// populatorHasAddedPods - checker for whether the populator has finished
// adding pods to the desiredStateOfWorld cache at least once after sources
// are all ready (before sources are ready, pods are probably missing)
//
// operationExecutor - used to trigger attach/detach/mount/unmount operations
// safely (prevents more than one operation from being triggered on the same
// volume)
//
// mounter - mounter passed in from kubelet, passed down unmount path
//
// hostutil - hostutil passed in from kubelet
//
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
		skippedDuringReconstruction:   map[v1.UniqueVolumeName]*globalVolumeInfo{},
		volumePluginMgr:               volumePluginMgr,
		kubeletPodsDir:                kubeletPodsDir,
		timeOfLastSync:                time.Time{},
		volumesFailedReconstruction:   make([]podVolume, 0),
		volumesNeedDevicePath:         make([]v1.UniqueVolumeName, 0),
		volumesNeedReportedInUse:      make([]v1.UniqueVolumeName, 0),
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
	skippedDuringReconstruction   map[v1.UniqueVolumeName]*globalVolumeInfo
	kubeletPodsDir                string
	timeOfLastSync                time.Time
	volumesFailedReconstruction   []podVolume
	volumesNeedDevicePath         []v1.UniqueVolumeName
	volumesNeedReportedInUse      []v1.UniqueVolumeName
}

func (rc *reconciler) Run(stopCh <-chan struct{}) {
	if utilfeature.DefaultFeatureGate.Enabled(features.SELinuxMountReadWriteOncePod) {
		rc.runNew(stopCh)
		return
	}

	rc.runOld(stopCh)
}

func (rc *reconciler) unmountVolumes() {
	// Ensure volumes that should be unmounted are unmounted.
	for _, mountedVolume := range rc.actualStateOfWorld.GetAllMountedVolumes() {
		if !rc.desiredStateOfWorld.PodExistsInVolume(mountedVolume.PodName, mountedVolume.VolumeName, mountedVolume.SELinuxMountContext) {
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
		volMounted, devicePath, err := rc.actualStateOfWorld.PodExistsInVolume(volumeToMount.PodName, volumeToMount.VolumeName, volumeToMount.PersistentVolumeSize, volumeToMount.SELinuxLabel)
		volumeToMount.DevicePath = devicePath
		if cache.IsSELinuxMountMismatchError(err) {
			// The volume is mounted, but with an unexpected SELinux context.
			// It will get unmounted in unmountVolumes / unmountDetachDevices and
			// then removed from actualStateOfWorld.
			rc.desiredStateOfWorld.AddErrorToPod(volumeToMount.PodName, err.Error())
			continue
		} else if cache.IsVolumeNotAttachedError(err) {
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
		if !rc.desiredStateOfWorld.VolumeExists(attachedVolume.VolumeName, attachedVolume.SELinuxMountContext) &&
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

// ignore nestedpendingoperations.IsAlreadyExists and exponentialbackoff.IsExponentialBackoff errors, they are expected.
func isExpectedError(err error) bool {
	return nestedpendingoperations.IsAlreadyExists(err) || exponentialbackoff.IsExponentialBackoff(err) || operationexecutor.IsMountFailedPreconditionError(err)
}
