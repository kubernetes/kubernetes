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

// Package operationexecutor implements interfaces that enable execution of
// attach, detach, mount, and unmount operations with a
// nestedpendingoperations so that more than one operation is never triggered
// on the same volume for the same pod.
package operationexecutor

import (
	"fmt"
	"strings"
	"time"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/client/record"
	kevents "k8s.io/kubernetes/pkg/kubelet/events"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/util/nestedpendingoperations"
	volumetypes "k8s.io/kubernetes/pkg/volume/util/types"
	"k8s.io/kubernetes/pkg/volume/util/volumehelper"
)

// OperationExecutor defines a set of operations for attaching, detaching,
// mounting, or unmounting a volume that are executed with a NewNestedPendingOperations which
// prevents more than one operation from being triggered on the same volume.
//
// These operations should be idempotent (for example, AttachVolume should
// still succeed if the volume is already attached to the node, etc.). However,
// they depend on the volume plugins to implement this behavior.
//
// Once an operation completes successfully, the actualStateOfWorld is updated
// to indicate the volume is attached/detached/mounted/unmounted.
//
// If the OperationExecutor fails to start the operation because, for example,
// an operation with the same UniqueVolumeName is already pending, a non-nil
// error is returned.
//
// Once the operation is started, since it is executed asynchronously,
// errors are simply logged and the goroutine is terminated without updating
// actualStateOfWorld (callers are responsible for retrying as needed).
//
// Some of these operations may result in calls to the API server; callers are
// responsible for rate limiting on errors.
type OperationExecutor interface {
	// AttachVolume attaches the volume to the node specified in volumeToAttach.
	// It then updates the actual state of the world to reflect that.
	AttachVolume(volumeToAttach VolumeToAttach, actualStateOfWorld ActualStateOfWorldAttacherUpdater) error

	// VerifyVolumesAreAttached verifies the given list of volumes to see whether they are still attached to the node.
	// If any volume is not attached right now, it will update the actual state of the world to reflect that.
	// Note that this operation could be operated concurrently with other attach/detach operations.
	// In theory (but very unlikely in practise), race condition among these operations might mark volume as detached
	// even if it is attached. But reconciler can correct this in a short period of time.
	VerifyVolumesAreAttached(AttachedVolumes []AttachedVolume, nodeName types.NodeName, actualStateOfWorld ActualStateOfWorldAttacherUpdater) error

	// DetachVolume detaches the volume from the node specified in
	// volumeToDetach, and updates the actual state of the world to reflect
	// that. If verifySafeToDetach is set, a call is made to the fetch the node
	// object and it is used to verify that the volume does not exist in Node's
	// Status.VolumesInUse list (operation fails with error if it is).
	DetachVolume(volumeToDetach AttachedVolume, verifySafeToDetach bool, actualStateOfWorld ActualStateOfWorldAttacherUpdater) error

	// MountVolume mounts the volume to the pod specified in volumeToMount.
	// Specifically it will:
	// * Wait for the device to finish attaching (for attachable volumes only).
	// * Mount device to global mount path (for attachable volumes only).
	// * Update actual state of world to reflect volume is globally mounted (for
	//   attachable volumes only).
	// * Mount the volume to the pod specific path.
	// * Update actual state of world to reflect volume is mounted to the pod
	//   path.
	MountVolume(waitForAttachTimeout time.Duration, volumeToMount VolumeToMount, actualStateOfWorld ActualStateOfWorldMounterUpdater) error

	// UnmountVolume unmounts the volume from the pod specified in
	// volumeToUnmount and updates the actual state of the world to reflect that.
	UnmountVolume(volumeToUnmount MountedVolume, actualStateOfWorld ActualStateOfWorldMounterUpdater) error

	// UnmountDevice unmounts the volumes global mount path from the device (for
	// attachable volumes only, freeing it for detach. It then updates the
	// actual state of the world to reflect that.
	UnmountDevice(deviceToDetach AttachedVolume, actualStateOfWorld ActualStateOfWorldMounterUpdater, mounter mount.Interface) error

	// VerifyControllerAttachedVolume checks if the specified volume is present
	// in the specified nodes AttachedVolumes Status field. It uses kubeClient
	// to fetch the node object.
	// If the volume is found, the actual state of the world is updated to mark
	// the volume as attached.
	// If the volume does not implement the attacher interface, it is assumed to
	// be attached and the actual state of the world is updated accordingly.
	// If the volume is not found or there is an error (fetching the node
	// object, for example) then an error is returned which triggers exponential
	// back off on retries.
	VerifyControllerAttachedVolume(volumeToMount VolumeToMount, nodeName types.NodeName, actualStateOfWorld ActualStateOfWorldAttacherUpdater) error

	// IsOperationPending returns true if an operation for the given volumeName and podName is pending,
	// otherwise it returns false
	IsOperationPending(volumeName api.UniqueVolumeName, podName volumetypes.UniquePodName) bool
}

// NewOperationExecutor returns a new instance of OperationExecutor.
func NewOperationExecutor(
	kubeClient internalclientset.Interface,
	volumePluginMgr *volume.VolumePluginMgr,
	recorder record.EventRecorder,
	checkNodeCapabilitiesBeforeMount bool) OperationExecutor {

	return &operationExecutor{
		kubeClient:      kubeClient,
		volumePluginMgr: volumePluginMgr,
		pendingOperations: nestedpendingoperations.NewNestedPendingOperations(
			true /* exponentialBackOffOnError */),
		recorder: recorder,
		checkNodeCapabilitiesBeforeMount: checkNodeCapabilitiesBeforeMount,
	}
}

// ActualStateOfWorldMounterUpdater defines a set of operations updating the actual
// state of the world cache after successful mount/unmount.
type ActualStateOfWorldMounterUpdater interface {
	// Marks the specified volume as mounted to the specified pod
	MarkVolumeAsMounted(podName volumetypes.UniquePodName, podUID types.UID, volumeName api.UniqueVolumeName, mounter volume.Mounter, outerVolumeSpecName string, volumeGidValue string) error

	// Marks the specified volume as unmounted from the specified pod
	MarkVolumeAsUnmounted(podName volumetypes.UniquePodName, volumeName api.UniqueVolumeName) error

	// Marks the specified volume as having been globally mounted.
	MarkDeviceAsMounted(volumeName api.UniqueVolumeName) error

	// Marks the specified volume as having its global mount unmounted.
	MarkDeviceAsUnmounted(volumeName api.UniqueVolumeName) error
}

// ActualStateOfWorldAttacherUpdater defines a set of operations updating the
// actual state of the world cache after successful attach/detach/mount/unmount.
type ActualStateOfWorldAttacherUpdater interface {
	// Marks the specified volume as attached to the specified node.  If the
	// volume name is supplied, that volume name will be used.  If not, the
	// volume name is computed using the result from querying the plugin.
	//
	// TODO: in the future, we should be able to remove the volumeName
	// argument to this method -- since it is used only for attachable
	// volumes.  See issue 29695.
	MarkVolumeAsAttached(volumeName api.UniqueVolumeName, volumeSpec *volume.Spec, nodeName types.NodeName, devicePath string) error

	// Marks the specified volume as detached from the specified node
	MarkVolumeAsDetached(volumeName api.UniqueVolumeName, nodeName types.NodeName)

	// Marks desire to detach the specified volume (remove the volume from the node's
	// volumesToReportedAsAttached list)
	RemoveVolumeFromReportAsAttached(volumeName api.UniqueVolumeName, nodeName types.NodeName) error

	// Unmarks the desire to detach for the specified volume (add the volume back to
	// the node's volumesToReportedAsAttached list)
	AddVolumeToReportAsAttached(volumeName api.UniqueVolumeName, nodeName types.NodeName)
}

// VolumeToAttach represents a volume that should be attached to a node.
type VolumeToAttach struct {
	// VolumeName is the unique identifier for the volume that should be
	// attached.
	VolumeName api.UniqueVolumeName

	// VolumeSpec is a volume spec containing the specification for the volume
	// that should be attached.
	VolumeSpec *volume.Spec

	// NodeName is the identifier for the node that the volume should be
	// attached to.
	NodeName types.NodeName

	// scheduledPods is a map containing the set of pods that reference this
	// volume and are scheduled to the underlying node. The key in the map is
	// the name of the pod and the value is a pod object containing more
	// information about the pod.
	ScheduledPods []*api.Pod
}

// VolumeToMount represents a volume that should be attached to this node and
// mounted to the PodName.
type VolumeToMount struct {
	// VolumeName is the unique identifier for the volume that should be
	// mounted.
	VolumeName api.UniqueVolumeName

	// PodName is the unique identifier for the pod that the volume should be
	// mounted to after it is attached.
	PodName volumetypes.UniquePodName

	// VolumeSpec is a volume spec containing the specification for the volume
	// that should be mounted. Used to create NewMounter. Used to generate
	// InnerVolumeSpecName.
	VolumeSpec *volume.Spec

	// outerVolumeSpecName is the podSpec.Volume[x].Name of the volume. If the
	// volume was referenced through a persistent volume claim, this contains
	// the podSpec.Volume[x].Name of the persistent volume claim.
	OuterVolumeSpecName string

	// Pod to mount the volume to. Used to create NewMounter.
	Pod *api.Pod

	// PluginIsAttachable indicates that the plugin for this volume implements
	// the volume.Attacher interface
	PluginIsAttachable bool

	// VolumeGidValue contains the value of the GID annotation, if present.
	VolumeGidValue string

	// DevicePath contains the path on the node where the volume is attached.
	// For non-attachable volumes this is empty.
	DevicePath string

	// ReportedInUse indicates that the volume was successfully added to the
	// VolumesInUse field in the node's status.
	ReportedInUse bool
}

// AttachedVolume represents a volume that is attached to a node.
type AttachedVolume struct {
	// VolumeName is the unique identifier for the volume that is attached.
	VolumeName api.UniqueVolumeName

	// VolumeSpec is the volume spec containing the specification for the
	// volume that is attached.
	VolumeSpec *volume.Spec

	// NodeName is the identifier for the node that the volume is attached to.
	NodeName types.NodeName

	// PluginIsAttachable indicates that the plugin for this volume implements
	// the volume.Attacher interface
	PluginIsAttachable bool

	// DevicePath contains the path on the node where the volume is attached.
	// For non-attachable volumes this is empty.
	DevicePath string
}

// MountedVolume represents a volume that has successfully been mounted to a pod.
type MountedVolume struct {
	// PodName is the unique identifier of the pod mounted to.
	PodName volumetypes.UniquePodName

	// VolumeName is the unique identifier of the volume mounted to the pod.
	VolumeName api.UniqueVolumeName

	// InnerVolumeSpecName is the volume.Spec.Name() of the volume. If the
	// volume was referenced through a persistent volume claims, this contains
	// the name of the bound persistent volume object.
	// It is the name that plugins use in their pod mount path, i.e.
	// /var/lib/kubelet/pods/{podUID}/volumes/{escapeQualifiedPluginName}/{innerVolumeSpecName}/
	// PVC example,
	//   apiVersion: v1
	//   kind: PersistentVolume
	//   metadata:
	//     name: pv0003				<- InnerVolumeSpecName
	//   spec:
	//     capacity:
	//       storage: 5Gi
	//     accessModes:
	//       - ReadWriteOnce
	//     persistentVolumeReclaimPolicy: Recycle
	//     nfs:
	//       path: /tmp
	//       server: 172.17.0.2
	// Non-PVC example:
	//   apiVersion: v1
	//   kind: Pod
	//   metadata:
	//     name: test-pd
	//   spec:
	//     containers:
	//     - image: gcr.io/google_containers/test-webserver
	//     	 name: test-container
	//     	 volumeMounts:
	//     	 - mountPath: /test-pd
	//     	   name: test-volume
	//     volumes:
	//     - name: test-volume			<- InnerVolumeSpecName
	//     	 gcePersistentDisk:
	//     	   pdName: my-data-disk
	//     	   fsType: ext4
	InnerVolumeSpecName string

	// outerVolumeSpecName is the podSpec.Volume[x].Name of the volume. If the
	// volume was referenced through a persistent volume claim, this contains
	// the podSpec.Volume[x].Name of the persistent volume claim.
	// PVC example:
	//   kind: Pod
	//   apiVersion: v1
	//   metadata:
	//     name: mypod
	//   spec:
	//     containers:
	//       - name: myfrontend
	//         image: dockerfile/nginx
	//         volumeMounts:
	//         - mountPath: "/var/www/html"
	//           name: mypd
	//     volumes:
	//       - name: mypd				<- OuterVolumeSpecName
	//         persistentVolumeClaim:
	//           claimName: myclaim
	// Non-PVC example:
	//   apiVersion: v1
	//   kind: Pod
	//   metadata:
	//     name: test-pd
	//   spec:
	//     containers:
	//     - image: gcr.io/google_containers/test-webserver
	//     	 name: test-container
	//     	 volumeMounts:
	//     	 - mountPath: /test-pd
	//     	   name: test-volume
	//     volumes:
	//     - name: test-volume			<- OuterVolumeSpecName
	//     	 gcePersistentDisk:
	//     	   pdName: my-data-disk
	//     	   fsType: ext4
	OuterVolumeSpecName string

	// PluginName is the "Unescaped Qualified" name of the volume plugin used to
	// mount and unmount this volume. It can be used to fetch the volume plugin
	// to unmount with, on demand. It is also the name that plugins use, though
	// escaped, in their pod mount path, i.e.
	// /var/lib/kubelet/pods/{podUID}/volumes/{escapeQualifiedPluginName}/{outerVolumeSpecName}/
	PluginName string

	// PodUID is the UID of the pod mounted to. It is also the string used by
	// plugins in their pod mount path, i.e.
	// /var/lib/kubelet/pods/{podUID}/volumes/{escapeQualifiedPluginName}/{outerVolumeSpecName}/
	PodUID types.UID

	// Mounter is the volume mounter used to mount this volume. It is required
	// by kubelet to create container.VolumeMap.
	Mounter volume.Mounter

	// VolumeGidValue contains the value of the GID annotation, if present.
	VolumeGidValue string
}

type operationExecutor struct {
	// Used to fetch objects from the API server like Node in the
	// VerifyControllerAttachedVolume operation.
	kubeClient internalclientset.Interface

	// volumePluginMgr is the volume plugin manager used to create volume
	// plugin objects.
	volumePluginMgr *volume.VolumePluginMgr

	// pendingOperations keeps track of pending attach and detach operations so
	// multiple operations are not started on the same volume
	pendingOperations nestedpendingoperations.NestedPendingOperations

	// recorder is used to record events in the API server
	recorder record.EventRecorder

	// checkNodeCapabilitiesBeforeMount, if set, enables the CanMount check,
	// which verifies that the components (binaries, etc.) required to mount
	// the volume are available on the underlying node before attempting mount.
	checkNodeCapabilitiesBeforeMount bool
}

func (oe *operationExecutor) IsOperationPending(volumeName api.UniqueVolumeName, podName volumetypes.UniquePodName) bool {
	return oe.pendingOperations.IsOperationPending(volumeName, podName)
}

func (oe *operationExecutor) AttachVolume(
	volumeToAttach VolumeToAttach,
	actualStateOfWorld ActualStateOfWorldAttacherUpdater) error {
	attachFunc, err :=
		oe.generateAttachVolumeFunc(volumeToAttach, actualStateOfWorld)
	if err != nil {
		return err
	}

	return oe.pendingOperations.Run(
		volumeToAttach.VolumeName, "" /* podName */, attachFunc)
}

func (oe *operationExecutor) DetachVolume(
	volumeToDetach AttachedVolume,
	verifySafeToDetach bool,
	actualStateOfWorld ActualStateOfWorldAttacherUpdater) error {
	detachFunc, err :=
		oe.generateDetachVolumeFunc(volumeToDetach, verifySafeToDetach, actualStateOfWorld)
	if err != nil {
		return err
	}

	return oe.pendingOperations.Run(
		volumeToDetach.VolumeName, "" /* podName */, detachFunc)
}

func (oe *operationExecutor) VerifyVolumesAreAttached(
	attachedVolumes []AttachedVolume,
	nodeName types.NodeName,
	actualStateOfWorld ActualStateOfWorldAttacherUpdater) error {
	volumesAreAttachedFunc, err :=
		oe.generateVolumesAreAttachedFunc(attachedVolumes, nodeName, actualStateOfWorld)
	if err != nil {
		return err
	}
	// Give an empty UniqueVolumeName so that this operation could be executed concurrently.
	return oe.pendingOperations.Run("" /* volumeName */, "" /* podName */, volumesAreAttachedFunc)
}

func (oe *operationExecutor) MountVolume(
	waitForAttachTimeout time.Duration,
	volumeToMount VolumeToMount,
	actualStateOfWorld ActualStateOfWorldMounterUpdater) error {
	mountFunc, err := oe.generateMountVolumeFunc(
		waitForAttachTimeout, volumeToMount, actualStateOfWorld)
	if err != nil {
		return err
	}

	podName := volumetypes.UniquePodName("")
	// TODO: remove this -- not necessary
	if !volumeToMount.PluginIsAttachable {
		// Non-attachable volume plugins can execute mount for multiple pods
		// referencing the same volume in parallel
		podName = volumehelper.GetUniquePodName(volumeToMount.Pod)
	}

	return oe.pendingOperations.Run(
		volumeToMount.VolumeName, podName, mountFunc)
}

func (oe *operationExecutor) UnmountVolume(
	volumeToUnmount MountedVolume,
	actualStateOfWorld ActualStateOfWorldMounterUpdater) error {

	unmountFunc, err :=
		oe.generateUnmountVolumeFunc(volumeToUnmount, actualStateOfWorld)
	if err != nil {
		return err
	}

	// All volume plugins can execute mount for multiple pods referencing the
	// same volume in parallel
	podName := volumetypes.UniquePodName(volumeToUnmount.PodUID)

	return oe.pendingOperations.Run(
		volumeToUnmount.VolumeName, podName, unmountFunc)
}

func (oe *operationExecutor) UnmountDevice(
	deviceToDetach AttachedVolume,
	actualStateOfWorld ActualStateOfWorldMounterUpdater,
	mounter mount.Interface) error {
	unmountDeviceFunc, err :=
		oe.generateUnmountDeviceFunc(deviceToDetach, actualStateOfWorld, mounter)
	if err != nil {
		return err
	}

	return oe.pendingOperations.Run(
		deviceToDetach.VolumeName, "" /* podName */, unmountDeviceFunc)
}

func (oe *operationExecutor) VerifyControllerAttachedVolume(
	volumeToMount VolumeToMount,
	nodeName types.NodeName,
	actualStateOfWorld ActualStateOfWorldAttacherUpdater) error {
	verifyControllerAttachedVolumeFunc, err :=
		oe.generateVerifyControllerAttachedVolumeFunc(volumeToMount, nodeName, actualStateOfWorld)
	if err != nil {
		return err
	}

	return oe.pendingOperations.Run(
		volumeToMount.VolumeName, "" /* podName */, verifyControllerAttachedVolumeFunc)
}

func (oe *operationExecutor) generateVolumesAreAttachedFunc(
	attachedVolumes []AttachedVolume,
	nodeName types.NodeName,
	actualStateOfWorld ActualStateOfWorldAttacherUpdater) (func() error, error) {

	// volumesPerPlugin maps from a volume plugin to a list of volume specs which belong
	// to this type of plugin
	volumesPerPlugin := make(map[string][]*volume.Spec)
	// volumeSpecMap maps from a volume spec to its unique volumeName which will be used
	// when calling MarkVolumeAsDetached
	volumeSpecMap := make(map[*volume.Spec]api.UniqueVolumeName)
	// Iterate each volume spec and put them into a map index by the pluginName
	for _, volumeAttached := range attachedVolumes {
		volumePlugin, err :=
			oe.volumePluginMgr.FindPluginBySpec(volumeAttached.VolumeSpec)
		if err != nil || volumePlugin == nil {
			glog.Errorf(
				"VolumesAreAttached.FindPluginBySpec failed for volume %q (spec.Name: %q) on node %q with error: %v",
				volumeAttached.VolumeName,
				volumeAttached.VolumeSpec.Name(),
				volumeAttached.NodeName,
				err)
		}
		volumeSpecList, pluginExists := volumesPerPlugin[volumePlugin.GetPluginName()]
		if !pluginExists {
			volumeSpecList = []*volume.Spec{}
		}
		volumeSpecList = append(volumeSpecList, volumeAttached.VolumeSpec)
		volumesPerPlugin[volumePlugin.GetPluginName()] = volumeSpecList
		volumeSpecMap[volumeAttached.VolumeSpec] = volumeAttached.VolumeName
	}

	return func() error {

		// For each volume plugin, pass the list of volume specs to VolumesAreAttached to check
		// whether the volumes are still attached.
		for pluginName, volumesSpecs := range volumesPerPlugin {
			attachableVolumePlugin, err :=
				oe.volumePluginMgr.FindAttachablePluginByName(pluginName)
			if err != nil || attachableVolumePlugin == nil {
				glog.Errorf(
					"VolumeAreAttached.FindAttachablePluginBySpec failed for plugin %q with: %v",
					pluginName,
					err)
				continue
			}

			volumeAttacher, newAttacherErr := attachableVolumePlugin.NewAttacher()
			if newAttacherErr != nil {
				glog.Errorf(
					"VolumesAreAttached failed for getting plugin %q with: %v",
					pluginName,
					newAttacherErr)
				continue
			}

			attached, areAttachedErr := volumeAttacher.VolumesAreAttached(volumesSpecs, nodeName)
			if areAttachedErr != nil {
				glog.Errorf(
					"VolumesAreAttached failed for checking on node %q with: %v",
					nodeName,
					areAttachedErr)
				continue
			}

			for spec, check := range attached {
				if !check {
					actualStateOfWorld.MarkVolumeAsDetached(volumeSpecMap[spec], nodeName)
					glog.V(1).Infof("VerifyVolumesAreAttached determined volume %q (spec.Name: %q) is no longer attached to node %q, therefore it was marked as detached.",
						volumeSpecMap[spec], spec.Name())
				}
			}
		}
		return nil
	}, nil
}

func (oe *operationExecutor) generateAttachVolumeFunc(
	volumeToAttach VolumeToAttach,
	actualStateOfWorld ActualStateOfWorldAttacherUpdater) (func() error, error) {
	// Get attacher plugin
	attachableVolumePlugin, err :=
		oe.volumePluginMgr.FindAttachablePluginBySpec(volumeToAttach.VolumeSpec)
	if err != nil || attachableVolumePlugin == nil {
		return nil, fmt.Errorf(
			"AttachVolume.FindAttachablePluginBySpec failed for volume %q (spec.Name: %q) from node %q with: %v",
			volumeToAttach.VolumeName,
			volumeToAttach.VolumeSpec.Name(),
			volumeToAttach.NodeName,
			err)
	}

	volumeAttacher, newAttacherErr := attachableVolumePlugin.NewAttacher()
	if newAttacherErr != nil {
		return nil, fmt.Errorf(
			"AttachVolume.NewAttacher failed for volume %q (spec.Name: %q) from node %q with: %v",
			volumeToAttach.VolumeName,
			volumeToAttach.VolumeSpec.Name(),
			volumeToAttach.NodeName,
			newAttacherErr)
	}

	return func() error {
		// Execute attach
		devicePath, attachErr := volumeAttacher.Attach(
			volumeToAttach.VolumeSpec, volumeToAttach.NodeName)

		if attachErr != nil {
			// On failure, return error. Caller will log and retry.
			err := fmt.Errorf(
				"Failed to attach volume %q on node %q with: %v",
				volumeToAttach.VolumeSpec.Name(),
				volumeToAttach.NodeName,
				attachErr)
			for _, pod := range volumeToAttach.ScheduledPods {
				oe.recorder.Eventf(pod, api.EventTypeWarning, kevents.FailedMountVolume, err.Error())
			}
			return err
		}

		glog.Infof(
			"AttachVolume.Attach succeeded for volume %q (spec.Name: %q) from node %q.",
			volumeToAttach.VolumeName,
			volumeToAttach.VolumeSpec.Name(),
			volumeToAttach.NodeName)

		// Update actual state of world
		addVolumeNodeErr := actualStateOfWorld.MarkVolumeAsAttached(
			api.UniqueVolumeName(""), volumeToAttach.VolumeSpec, volumeToAttach.NodeName, devicePath)
		if addVolumeNodeErr != nil {
			// On failure, return error. Caller will log and retry.
			return fmt.Errorf(
				"AttachVolume.MarkVolumeAsAttached failed for volume %q (spec.Name: %q) from node %q with: %v.",
				volumeToAttach.VolumeName,
				volumeToAttach.VolumeSpec.Name(),
				volumeToAttach.NodeName,
				addVolumeNodeErr)
		}

		return nil
	}, nil
}

func (oe *operationExecutor) generateDetachVolumeFunc(
	volumeToDetach AttachedVolume,
	verifySafeToDetach bool,
	actualStateOfWorld ActualStateOfWorldAttacherUpdater) (func() error, error) {
	// Get attacher plugin
	attachableVolumePlugin, err :=
		oe.volumePluginMgr.FindAttachablePluginBySpec(volumeToDetach.VolumeSpec)
	if err != nil || attachableVolumePlugin == nil {
		return nil, fmt.Errorf(
			"DetachVolume.FindAttachablePluginBySpec failed for volume %q (spec.Name: %q) from node %q with: %v",
			volumeToDetach.VolumeName,
			volumeToDetach.VolumeSpec.Name(),
			volumeToDetach.NodeName,
			err)
	}

	volumeName, err :=
		attachableVolumePlugin.GetVolumeName(volumeToDetach.VolumeSpec)
	if err != nil {
		return nil, fmt.Errorf(
			"DetachVolume.GetVolumeName failed for volume %q (spec.Name: %q) from node %q with: %v",
			volumeToDetach.VolumeName,
			volumeToDetach.VolumeSpec.Name(),
			volumeToDetach.NodeName,
			err)
	}

	volumeDetacher, err := attachableVolumePlugin.NewDetacher()
	if err != nil {
		return nil, fmt.Errorf(
			"DetachVolume.NewDetacher failed for volume %q (spec.Name: %q) from node %q with: %v",
			volumeToDetach.VolumeName,
			volumeToDetach.VolumeSpec.Name(),
			volumeToDetach.NodeName,
			err)
	}

	return func() error {
		var err error
		if verifySafeToDetach {
			err = oe.verifyVolumeIsSafeToDetach(volumeToDetach)
		}
		if err == nil {
			err = volumeDetacher.Detach(volumeName, volumeToDetach.NodeName)
		}
		if err != nil {
			// On failure, add volume back to ReportAsAttached list
			actualStateOfWorld.AddVolumeToReportAsAttached(
				volumeToDetach.VolumeName, volumeToDetach.NodeName)
			return fmt.Errorf(
				"DetachVolume.Detach failed for volume %q (spec.Name: %q) from node %q with: %v",
				volumeToDetach.VolumeName,
				volumeToDetach.VolumeSpec.Name(),
				volumeToDetach.NodeName,
				err)
		}

		glog.Infof(
			"DetachVolume.Detach succeeded for volume %q (spec.Name: %q) from node %q.",
			volumeToDetach.VolumeName,
			volumeToDetach.VolumeSpec.Name(),
			volumeToDetach.NodeName)

		// Update actual state of world
		actualStateOfWorld.MarkVolumeAsDetached(
			volumeToDetach.VolumeName, volumeToDetach.NodeName)

		return nil
	}, nil
}

func (oe *operationExecutor) verifyVolumeIsSafeToDetach(
	volumeToDetach AttachedVolume) error {
	// Fetch current node object
	node, fetchErr := oe.kubeClient.Core().Nodes().Get(string(volumeToDetach.NodeName))
	if fetchErr != nil {
		if errors.IsNotFound(fetchErr) {
			glog.Warningf("Node %q not found on API server. DetachVolume will skip safe to detach check.",
				volumeToDetach.NodeName,
				volumeToDetach.VolumeName,
				volumeToDetach.VolumeSpec.Name())
			return nil
		}

		// On failure, return error. Caller will log and retry.
		return fmt.Errorf(
			"DetachVolume failed fetching node from API server for volume %q (spec.Name: %q) from node %q with: %v",
			volumeToDetach.VolumeName,
			volumeToDetach.VolumeSpec.Name(),
			volumeToDetach.NodeName,
			fetchErr)
	}

	if node == nil {
		// On failure, return error. Caller will log and retry.
		return fmt.Errorf(
			"DetachVolume failed fetching node from API server for volume %q (spec.Name: %q) from node %q. Error: node object retrieved from API server is nil.",
			volumeToDetach.VolumeName,
			volumeToDetach.VolumeSpec.Name(),
			volumeToDetach.NodeName)
	}

	for _, inUseVolume := range node.Status.VolumesInUse {
		if inUseVolume == volumeToDetach.VolumeName {
			return fmt.Errorf("DetachVolume failed for volume %q (spec.Name: %q) from node %q. Error: volume is still in use by node, according to Node status.",
				volumeToDetach.VolumeName,
				volumeToDetach.VolumeSpec.Name(),
				volumeToDetach.NodeName)
		}
	}

	// Volume is not marked as in use by node
	glog.Infof("Verified volume is safe to detach for volume %q (spec.Name: %q) from node %q.",
		volumeToDetach.VolumeName,
		volumeToDetach.VolumeSpec.Name(),
		volumeToDetach.NodeName)
	return nil
}

func (oe *operationExecutor) generateMountVolumeFunc(
	waitForAttachTimeout time.Duration,
	volumeToMount VolumeToMount,
	actualStateOfWorld ActualStateOfWorldMounterUpdater) (func() error, error) {
	// Get mounter plugin
	volumePlugin, err :=
		oe.volumePluginMgr.FindPluginBySpec(volumeToMount.VolumeSpec)
	if err != nil || volumePlugin == nil {
		return nil, fmt.Errorf(
			"MountVolume.FindPluginBySpec failed for volume %q (spec.Name: %q) pod %q (UID: %q) with: %v",
			volumeToMount.VolumeName,
			volumeToMount.VolumeSpec.Name(),
			volumeToMount.PodName,
			volumeToMount.Pod.UID,
			err)
	}

	volumeMounter, newMounterErr := volumePlugin.NewMounter(
		volumeToMount.VolumeSpec,
		volumeToMount.Pod,
		volume.VolumeOptions{})
	if newMounterErr != nil {
		return nil, fmt.Errorf(
			"MountVolume.NewMounter failed for volume %q (spec.Name: %q) pod %q (UID: %q) with: %v",
			volumeToMount.VolumeName,
			volumeToMount.VolumeSpec.Name(),
			volumeToMount.PodName,
			volumeToMount.Pod.UID,
			newMounterErr)
	}

	// Get attacher, if possible
	attachableVolumePlugin, _ :=
		oe.volumePluginMgr.FindAttachablePluginBySpec(volumeToMount.VolumeSpec)
	var volumeAttacher volume.Attacher
	if attachableVolumePlugin != nil {
		volumeAttacher, _ = attachableVolumePlugin.NewAttacher()
	}

	var fsGroup *int64
	if volumeToMount.Pod.Spec.SecurityContext != nil &&
		volumeToMount.Pod.Spec.SecurityContext.FSGroup != nil {
		fsGroup = volumeToMount.Pod.Spec.SecurityContext.FSGroup
	}

	return func() error {
		if volumeAttacher != nil {
			// Wait for attachable volumes to finish attaching
			glog.Infof(
				"Entering MountVolume.WaitForAttach for volume %q (spec.Name: %q) pod %q (UID: %q) DevicePath: %q",
				volumeToMount.VolumeName,
				volumeToMount.VolumeSpec.Name(),
				volumeToMount.PodName,
				volumeToMount.Pod.UID,
				volumeToMount.DevicePath)

			devicePath, err := volumeAttacher.WaitForAttach(
				volumeToMount.VolumeSpec, volumeToMount.DevicePath, waitForAttachTimeout)
			if err != nil {
				// On failure, return error. Caller will log and retry.
				return fmt.Errorf(
					"MountVolume.WaitForAttach failed for volume %q (spec.Name: %q) pod %q (UID: %q) with: %v",
					volumeToMount.VolumeName,
					volumeToMount.VolumeSpec.Name(),
					volumeToMount.PodName,
					volumeToMount.Pod.UID,
					err)
			}

			glog.Infof(
				"MountVolume.WaitForAttach succeeded for volume %q (spec.Name: %q) pod %q (UID: %q).",
				volumeToMount.VolumeName,
				volumeToMount.VolumeSpec.Name(),
				volumeToMount.PodName,
				volumeToMount.Pod.UID)

			deviceMountPath, err :=
				volumeAttacher.GetDeviceMountPath(volumeToMount.VolumeSpec)
			if err != nil {
				// On failure, return error. Caller will log and retry.
				return fmt.Errorf(
					"MountVolume.GetDeviceMountPath failed for volume %q (spec.Name: %q) pod %q (UID: %q) with: %v",
					volumeToMount.VolumeName,
					volumeToMount.VolumeSpec.Name(),
					volumeToMount.PodName,
					volumeToMount.Pod.UID,
					err)
			}

			// Mount device to global mount path
			err = volumeAttacher.MountDevice(
				volumeToMount.VolumeSpec,
				devicePath,
				deviceMountPath)
			if err != nil {
				// On failure, return error. Caller will log and retry.
				err := fmt.Errorf(
					"MountVolume.MountDevice failed for volume %q (spec.Name: %q) pod %q (UID: %q) with: %v",
					volumeToMount.VolumeName,
					volumeToMount.VolumeSpec.Name(),
					volumeToMount.PodName,
					volumeToMount.Pod.UID,
					err)
				oe.recorder.Eventf(volumeToMount.Pod, api.EventTypeWarning, kevents.FailedMountVolume, err.Error())
				return err
			}

			glog.Infof(
				"MountVolume.MountDevice succeeded for volume %q (spec.Name: %q) pod %q (UID: %q) device mount path %q",
				volumeToMount.VolumeName,
				volumeToMount.VolumeSpec.Name(),
				volumeToMount.PodName,
				volumeToMount.Pod.UID,
				deviceMountPath)

			// Update actual state of world to reflect volume is globally mounted
			markDeviceMountedErr := actualStateOfWorld.MarkDeviceAsMounted(
				volumeToMount.VolumeName)
			if markDeviceMountedErr != nil {
				// On failure, return error. Caller will log and retry.
				return fmt.Errorf(
					"MountVolume.MarkDeviceAsMounted failed for volume %q (spec.Name: %q) pod %q (UID: %q) with: %v",
					volumeToMount.VolumeName,
					volumeToMount.VolumeSpec.Name(),
					volumeToMount.PodName,
					volumeToMount.Pod.UID,
					markDeviceMountedErr)
			}
		}

		if oe.checkNodeCapabilitiesBeforeMount {
			if canMountErr := volumeMounter.CanMount(); canMountErr != nil {
				errMsg := fmt.Sprintf("Unable to mount volume %v (spec.Name: %v) on pod %v (UID: %v). Verify that your node machine has the required components before attempting to mount this volume type. %s", volumeToMount.VolumeName, volumeToMount.VolumeSpec.Name(), volumeToMount.Pod.Name, volumeToMount.Pod.UID, canMountErr.Error())
				oe.recorder.Eventf(volumeToMount.Pod, api.EventTypeWarning, kevents.FailedMountVolume, errMsg)
				glog.Errorf(errMsg)
				return fmt.Errorf(errMsg)
			}
		}

		// Execute mount
		mountErr := volumeMounter.SetUp(fsGroup)
		if mountErr != nil {
			// On failure, return error. Caller will log and retry.
			err := fmt.Errorf(
				"MountVolume.SetUp failed for volume %q (spec.Name: %q) pod %q (UID: %q) with: %v",
				volumeToMount.VolumeName,
				volumeToMount.VolumeSpec.Name(),
				volumeToMount.PodName,
				volumeToMount.Pod.UID,
				mountErr)
			oe.recorder.Eventf(volumeToMount.Pod, api.EventTypeWarning, kevents.FailedMountVolume, err.Error())
			return err
		}

		glog.Infof(
			"MountVolume.SetUp succeeded for volume %q (spec.Name: %q) pod %q (UID: %q).",
			volumeToMount.VolumeName,
			volumeToMount.VolumeSpec.Name(),
			volumeToMount.PodName,
			volumeToMount.Pod.UID)

		// Update actual state of world
		markVolMountedErr := actualStateOfWorld.MarkVolumeAsMounted(
			volumeToMount.PodName,
			volumeToMount.Pod.UID,
			volumeToMount.VolumeName,
			volumeMounter,
			volumeToMount.OuterVolumeSpecName,
			volumeToMount.VolumeGidValue)
		if markVolMountedErr != nil {
			// On failure, return error. Caller will log and retry.
			return fmt.Errorf(
				"MountVolume.MarkVolumeAsMounted failed for volume %q (spec.Name: %q) pod %q (UID: %q) with: %v",
				volumeToMount.VolumeName,
				volumeToMount.VolumeSpec.Name(),
				volumeToMount.PodName,
				volumeToMount.Pod.UID,
				markVolMountedErr)
		}

		return nil
	}, nil
}

func (oe *operationExecutor) generateUnmountVolumeFunc(
	volumeToUnmount MountedVolume,
	actualStateOfWorld ActualStateOfWorldMounterUpdater) (func() error, error) {
	// Get mountable plugin
	volumePlugin, err :=
		oe.volumePluginMgr.FindPluginByName(volumeToUnmount.PluginName)
	if err != nil || volumePlugin == nil {
		return nil, fmt.Errorf(
			"UnmountVolume.FindPluginByName failed for volume %q (volume.spec.Name: %q) pod %q (UID: %q) err=%v",
			volumeToUnmount.VolumeName,
			volumeToUnmount.OuterVolumeSpecName,
			volumeToUnmount.PodName,
			volumeToUnmount.PodUID,
			err)
	}

	volumeUnmounter, newUnmounterErr := volumePlugin.NewUnmounter(
		volumeToUnmount.InnerVolumeSpecName, volumeToUnmount.PodUID)
	if newUnmounterErr != nil {
		return nil, fmt.Errorf(
			"UnmountVolume.NewUnmounter failed for volume %q (volume.spec.Name: %q) pod %q (UID: %q) err=%v",
			volumeToUnmount.VolumeName,
			volumeToUnmount.OuterVolumeSpecName,
			volumeToUnmount.PodName,
			volumeToUnmount.PodUID,
			newUnmounterErr)
	}

	return func() error {
		// Execute unmount
		unmountErr := volumeUnmounter.TearDown()
		if unmountErr != nil {
			// On failure, return error. Caller will log and retry.
			return fmt.Errorf(
				"UnmountVolume.TearDown failed for volume %q (volume.spec.Name: %q) pod %q (UID: %q) with: %v",
				volumeToUnmount.VolumeName,
				volumeToUnmount.OuterVolumeSpecName,
				volumeToUnmount.PodName,
				volumeToUnmount.PodUID,
				unmountErr)
		}

		glog.Infof(
			"UnmountVolume.TearDown succeeded for volume %q (OuterVolumeSpecName: %q) pod %q (UID: %q). InnerVolumeSpecName %q. PluginName %q, VolumeGidValue %q",
			volumeToUnmount.VolumeName,
			volumeToUnmount.OuterVolumeSpecName,
			volumeToUnmount.PodName,
			volumeToUnmount.PodUID,
			volumeToUnmount.InnerVolumeSpecName,
			volumeToUnmount.PluginName,
			volumeToUnmount.VolumeGidValue)

		// Update actual state of world
		markVolMountedErr := actualStateOfWorld.MarkVolumeAsUnmounted(
			volumeToUnmount.PodName, volumeToUnmount.VolumeName)
		if markVolMountedErr != nil {
			// On failure, just log and exit
			glog.Errorf(
				"UnmountVolume.MarkVolumeAsUnmounted failed for volume %q (volume.spec.Name: %q) pod %q (UID: %q) with: %v",
				volumeToUnmount.VolumeName,
				volumeToUnmount.OuterVolumeSpecName,
				volumeToUnmount.PodName,
				volumeToUnmount.PodUID,
				markVolMountedErr)
		}

		return nil
	}, nil
}

func (oe *operationExecutor) generateUnmountDeviceFunc(
	deviceToDetach AttachedVolume,
	actualStateOfWorld ActualStateOfWorldMounterUpdater,
	mounter mount.Interface) (func() error, error) {
	// Get attacher plugin
	attachableVolumePlugin, err :=
		oe.volumePluginMgr.FindAttachablePluginBySpec(deviceToDetach.VolumeSpec)
	if err != nil || attachableVolumePlugin == nil {
		return nil, fmt.Errorf(
			"UnmountDevice.FindAttachablePluginBySpec failed for volume %q (spec.Name: %q) with: %v",
			deviceToDetach.VolumeName,
			deviceToDetach.VolumeSpec.Name(),
			err)
	}

	volumeDetacher, err := attachableVolumePlugin.NewDetacher()
	if err != nil {
		return nil, fmt.Errorf(
			"UnmountDevice.NewDetacher failed for volume %q (spec.Name: %q) with: %v",
			deviceToDetach.VolumeName,
			deviceToDetach.VolumeSpec.Name(),
			err)
	}

	volumeAttacher, err := attachableVolumePlugin.NewAttacher()
	if err != nil {
		return nil, fmt.Errorf(
			"UnmountDevice.NewAttacher failed for volume %q (spec.Name: %q) with: %v",
			deviceToDetach.VolumeName,
			deviceToDetach.VolumeSpec.Name(),
			err)
	}

	return func() error {
		deviceMountPath, err :=
			volumeAttacher.GetDeviceMountPath(deviceToDetach.VolumeSpec)
		if err != nil {
			// On failure, return error. Caller will log and retry.
			return fmt.Errorf(
				"GetDeviceMountPath failed for volume %q (spec.Name: %q) with: %v",
				deviceToDetach.VolumeName,
				deviceToDetach.VolumeSpec.Name(),
				err)
		}
		refs, err := attachableVolumePlugin.GetDeviceMountRefs(deviceMountPath)

		if err != nil || hasMountRefs(deviceMountPath, refs) {
			if err == nil {
				err = fmt.Errorf("The device mount path %q is still mounted by other references %v", deviceMountPath, refs)
			}
			return fmt.Errorf(
				"GetDeviceMountRefs check failed for volume %q (spec.Name: %q) with: %v",
				deviceToDetach.VolumeName,
				deviceToDetach.VolumeSpec.Name(),
				err)
		}
		// Execute unmount
		unmountDeviceErr := volumeDetacher.UnmountDevice(deviceMountPath)
		if unmountDeviceErr != nil {
			// On failure, return error. Caller will log and retry.
			return fmt.Errorf(
				"UnmountDevice failed for volume %q (spec.Name: %q) with: %v",
				deviceToDetach.VolumeName,
				deviceToDetach.VolumeSpec.Name(),
				unmountDeviceErr)
		}
		// Before logging that UnmountDevice succeeded and moving on,
		// use mounter.PathIsDevice to check if the path is a device,
		// if so use mounter.DeviceOpened to check if the device is in use anywhere
		// else on the system. Retry if it returns true.
		isDevicePath, devicePathErr := mounter.PathIsDevice(deviceToDetach.DevicePath)
		var deviceOpened bool
		var deviceOpenedErr error
		if !isDevicePath && devicePathErr == nil {
			// not a device path or path doesn't exist
			//TODO: refer to #36092
			glog.V(3).Infof("Not checking device path %s", deviceToDetach.DevicePath)
			deviceOpened = false
		} else {
			deviceOpened, deviceOpenedErr = mounter.DeviceOpened(deviceToDetach.DevicePath)
			if deviceOpenedErr != nil {
				return fmt.Errorf(
					"UnmountDevice.DeviceOpened failed for volume %q (spec.Name: %q) with: %v",
					deviceToDetach.VolumeName,
					deviceToDetach.VolumeSpec.Name(),
					deviceOpenedErr)
			}
		}
		// The device is still in use elsewhere. Caller will log and retry.
		if deviceOpened {
			return fmt.Errorf(
				"UnmountDevice failed for volume %q (spec.Name: %q) because the device is in use when it was no longer expected to be in use",
				deviceToDetach.VolumeName,
				deviceToDetach.VolumeSpec.Name())
		}

		glog.Infof(
			"UnmountDevice succeeded for volume %q (spec.Name: %q).",
			deviceToDetach.VolumeName,
			deviceToDetach.VolumeSpec.Name())

		// Update actual state of world
		markDeviceUnmountedErr := actualStateOfWorld.MarkDeviceAsUnmounted(
			deviceToDetach.VolumeName)
		if markDeviceUnmountedErr != nil {
			// On failure, return error. Caller will log and retry.
			return fmt.Errorf(
				"MarkDeviceAsUnmounted failed for device %q (spec.Name: %q) with: %v",
				deviceToDetach.VolumeName,
				deviceToDetach.VolumeSpec.Name(),
				markDeviceUnmountedErr)
		}

		return nil
	}, nil
}

// TODO: this is a workaround for the unmount device issue caused by gci mounter.
// In GCI cluster, if gci mounter is used for mounting, the container started by mounter
// script will cause additional mounts created in the container. Since these mounts are
// irrelavant to the original mounts, they should be not considered when checking the
// mount references. Current solution is to filter out those mount paths that contain
// the string of original mount path.
// Plan to work on better approach to solve this issue.

func hasMountRefs(mountPath string, mountRefs []string) bool {
	count := 0
	for _, ref := range mountRefs {
		if !strings.Contains(ref, mountPath) {
			count = count + 1
		}
	}
	return count > 0
}

func (oe *operationExecutor) generateVerifyControllerAttachedVolumeFunc(
	volumeToMount VolumeToMount,
	nodeName types.NodeName,
	actualStateOfWorld ActualStateOfWorldAttacherUpdater) (func() error, error) {
	return func() error {
		if !volumeToMount.PluginIsAttachable {
			// If the volume does not implement the attacher interface, it is
			// assumed to be attached and the actual state of the world is
			// updated accordingly.

			addVolumeNodeErr := actualStateOfWorld.MarkVolumeAsAttached(
				volumeToMount.VolumeName, volumeToMount.VolumeSpec, nodeName, "" /* devicePath */)
			if addVolumeNodeErr != nil {
				// On failure, return error. Caller will log and retry.
				return fmt.Errorf(
					"VerifyControllerAttachedVolume.MarkVolumeAsAttachedByUniqueVolumeName failed for volume %q (spec.Name: %q) pod %q (UID: %q) with: %v.",
					volumeToMount.VolumeName,
					volumeToMount.VolumeSpec.Name(),
					volumeToMount.PodName,
					volumeToMount.Pod.UID,
					addVolumeNodeErr)
			}

			return nil
		}

		if !volumeToMount.ReportedInUse {
			// If the given volume has not yet been added to the list of
			// VolumesInUse in the node's volume status, do not proceed, return
			// error. Caller will log and retry. The node status is updated
			// periodically by kubelet, so it may take as much as 10 seconds
			// before this clears.
			// Issue #28141 to enable on demand status updates.
			return fmt.Errorf("Volume %q (spec.Name: %q) pod %q (UID: %q) has not yet been added to the list of VolumesInUse in the node's volume status.",
				volumeToMount.VolumeName,
				volumeToMount.VolumeSpec.Name(),
				volumeToMount.PodName,
				volumeToMount.Pod.UID)
		}

		// Fetch current node object
		node, fetchErr := oe.kubeClient.Core().Nodes().Get(string(nodeName))
		if fetchErr != nil {
			// On failure, return error. Caller will log and retry.
			return fmt.Errorf(
				"VerifyControllerAttachedVolume failed fetching node from API server. Volume %q (spec.Name: %q) pod %q (UID: %q). Error: %v.",
				volumeToMount.VolumeName,
				volumeToMount.VolumeSpec.Name(),
				volumeToMount.PodName,
				volumeToMount.Pod.UID,
				fetchErr)
		}

		if node == nil {
			// On failure, return error. Caller will log and retry.
			return fmt.Errorf(
				"VerifyControllerAttachedVolume failed. Volume %q (spec.Name: %q) pod %q (UID: %q). Error: node object retrieved from API server is nil.",
				volumeToMount.VolumeName,
				volumeToMount.VolumeSpec.Name(),
				volumeToMount.PodName,
				volumeToMount.Pod.UID)
		}

		for _, attachedVolume := range node.Status.VolumesAttached {
			if attachedVolume.Name == volumeToMount.VolumeName {
				addVolumeNodeErr := actualStateOfWorld.MarkVolumeAsAttached(
					api.UniqueVolumeName(""), volumeToMount.VolumeSpec, nodeName, attachedVolume.DevicePath)
				glog.Infof("Controller successfully attached volume %q (spec.Name: %q) pod %q (UID: %q) devicePath: %q",
					volumeToMount.VolumeName,
					volumeToMount.VolumeSpec.Name(),
					volumeToMount.PodName,
					volumeToMount.Pod.UID,
					attachedVolume.DevicePath)

				if addVolumeNodeErr != nil {
					// On failure, return error. Caller will log and retry.
					return fmt.Errorf(
						"VerifyControllerAttachedVolume.MarkVolumeAsAttached failed for volume %q (spec.Name: %q) pod %q (UID: %q) with: %v.",
						volumeToMount.VolumeName,
						volumeToMount.VolumeSpec.Name(),
						volumeToMount.PodName,
						volumeToMount.Pod.UID,
						addVolumeNodeErr)
				}
				return nil
			}
		}

		// Volume not attached, return error. Caller will log and retry.
		return fmt.Errorf("Volume %q (spec.Name: %q) pod %q (UID: %q) is not yet attached according to node status.",
			volumeToMount.VolumeName,
			volumeToMount.VolumeSpec.Name(),
			volumeToMount.PodName,
			volumeToMount.Pod.UID)
	}, nil
}
