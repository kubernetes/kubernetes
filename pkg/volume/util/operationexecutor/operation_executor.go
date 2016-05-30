/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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
// attach, detach, mount, and unmount operations with a goroutinemap so that
// more than one operation is never triggered on the same volume.
package operationexecutor

import (
	"fmt"
	"time"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/goroutinemap"
	"k8s.io/kubernetes/pkg/volume"
	volumetypes "k8s.io/kubernetes/pkg/volume/util/types"
)

// OperationExecutor defines a set of operations for attaching, detaching,
// mounting, or unmounting a volume that are executed with a goroutinemap which
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
type OperationExecutor interface {
	// AttachVolume attaches the volume to the node specified in volumeToAttach.
	// It then updates the actual state of the world to reflect that.
	AttachVolume(volumeToAttach VolumeToAttach, actualStateOfWorld ActualStateOfWorldAttacherUpdater) error

	// DetachVolume detaches the volume from the node specified in
	// volumeToDetach, and updates the actual state of the world to reflect
	// that.
	DetachVolume(volumeToDetach AttachedVolume, actualStateOfWorld ActualStateOfWorldAttacherUpdater) error

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
	UnmountDevice(deviceToDetach AttachedVolume, actualStateOfWorld ActualStateOfWorldMounterUpdater) error
}

// NewOperationExecutor returns a new instance of OperationExecutor.
func NewOperationExecutor(
	volumePluginMgr *volume.VolumePluginMgr) OperationExecutor {
	return &operationExecutor{
		volumePluginMgr:   volumePluginMgr,
		pendingOperations: goroutinemap.NewGoRoutineMap(),
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
	// Marks the specified volume as attached to the specified node
	MarkVolumeAsAttached(volumeSpec *volume.Spec, nodeName string) error

	// Marks the specified volume as detached from the specified node
	MarkVolumeAsDetached(volumeName api.UniqueVolumeName, nodeName string)
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
	NodeName string
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
}

// AttachedVolume represents a volume that is attached to a node.
type AttachedVolume struct {
	// VolumeName is the unique identifier for the volume that is attached.
	VolumeName api.UniqueVolumeName

	// VolumeSpec is the volume spec containing the specification for the
	// volume that is attached.
	VolumeSpec *volume.Spec

	// NodeName is the identifier for the node that the volume is attached to.
	NodeName string

	// PluginIsAttachable indicates that the plugin for this volume implements
	// the volume.Attacher interface
	PluginIsAttachable bool
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
	// volumePluginMgr is the volume plugin manager used to create volume
	// plugin objects.
	volumePluginMgr *volume.VolumePluginMgr
	// pendingOperations keeps track of pending attach and detach operations so
	// multiple operations are not started on the same volume
	pendingOperations goroutinemap.GoRoutineMap
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
		string(volumeToAttach.VolumeName), attachFunc)
}

func (oe *operationExecutor) DetachVolume(
	volumeToDetach AttachedVolume,
	actualStateOfWorld ActualStateOfWorldAttacherUpdater) error {
	detachFunc, err :=
		oe.generateDetachVolumeFunc(volumeToDetach, actualStateOfWorld)
	if err != nil {
		return err
	}

	return oe.pendingOperations.Run(
		string(volumeToDetach.VolumeName), detachFunc)
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

	return oe.pendingOperations.Run(
		string(volumeToMount.VolumeName), mountFunc)
}

func (oe *operationExecutor) UnmountVolume(
	volumeToUnmount MountedVolume,
	actualStateOfWorld ActualStateOfWorldMounterUpdater) error {
	unmountFunc, err :=
		oe.generateUnmountVolumeFunc(volumeToUnmount, actualStateOfWorld)
	if err != nil {
		return err
	}

	return oe.pendingOperations.Run(
		string(volumeToUnmount.VolumeName), unmountFunc)
}

func (oe *operationExecutor) UnmountDevice(
	deviceToDetach AttachedVolume,
	actualStateOfWorld ActualStateOfWorldMounterUpdater) error {
	unmountDeviceFunc, err :=
		oe.generateUnmountDeviceFunc(deviceToDetach, actualStateOfWorld)
	if err != nil {
		return err
	}

	return oe.pendingOperations.Run(
		string(deviceToDetach.VolumeName), unmountDeviceFunc)
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
		attachErr := volumeAttacher.Attach(
			volumeToAttach.VolumeSpec, volumeToAttach.NodeName)

		if attachErr != nil {
			// On failure, just log and exit. The controller will retry
			glog.Errorf(
				"AttachVolume.Attach failed for volume %q (spec.Name: %q) from node %q with: %v",
				volumeToAttach.VolumeName,
				volumeToAttach.VolumeSpec.Name(),
				volumeToAttach.NodeName,
				attachErr)
			return attachErr
		}

		glog.Infof(
			"AttachVolume.Attach succeeded for volume %q (spec.Name: %q) from node %q.",
			volumeToAttach.VolumeName,
			volumeToAttach.VolumeSpec.Name(),
			volumeToAttach.NodeName)

		// Update actual state of world
		addVolumeNodeErr := actualStateOfWorld.MarkVolumeAsAttached(
			volumeToAttach.VolumeSpec, volumeToAttach.NodeName)
		if addVolumeNodeErr != nil {
			// On failure, just log and exit. The controller will retry
			glog.Errorf(
				"AttachVolume.MarkVolumeAsAttached failed for volume %q (spec.Name: %q) from node %q with: %v.",
				volumeToAttach.VolumeName,
				volumeToAttach.VolumeSpec.Name(),
				volumeToAttach.NodeName,
				addVolumeNodeErr)
			return addVolumeNodeErr
		}

		return nil
	}, nil
}

func (oe *operationExecutor) generateDetachVolumeFunc(
	volumeToDetach AttachedVolume,
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
		// Execute detach
		detachErr := volumeDetacher.Detach(volumeName, volumeToDetach.NodeName)
		if detachErr != nil {
			// On failure, just log and exit. The controller will retry
			glog.Errorf(
				"DetachVolume.Detach failed for volume %q (spec.Name: %q) from node %q with: %v",
				volumeToDetach.VolumeName,
				volumeToDetach.VolumeSpec.Name(),
				volumeToDetach.NodeName,
				detachErr)
			return detachErr
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
				"Entering MountVolume.WaitForAttach for volume %q (spec.Name: %q) pod %q (UID: %q).",
				volumeToMount.VolumeName,
				volumeToMount.VolumeSpec.Name(),
				volumeToMount.PodName,
				volumeToMount.Pod.UID)

			devicePath, err := volumeAttacher.WaitForAttach(
				volumeToMount.VolumeSpec, waitForAttachTimeout)
			if err != nil {
				glog.Errorf(
					"MountVolume.WaitForAttach failed for volume %q (spec.Name: %q) pod %q (UID: %q) with: %v",
					volumeToMount.VolumeName,
					volumeToMount.VolumeSpec.Name(),
					volumeToMount.PodName,
					volumeToMount.Pod.UID,
					err)
				return err
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
				glog.Errorf(
					"MountVolume.GetDeviceMountPath failed for volume %q (spec.Name: %q) pod %q (UID: %q) with: %v",
					volumeToMount.VolumeName,
					volumeToMount.VolumeSpec.Name(),
					volumeToMount.PodName,
					volumeToMount.Pod.UID,
					err)
				return err
			}

			// Mount device to global mount path
			err = volumeAttacher.MountDevice(
				volumeToMount.VolumeSpec,
				devicePath,
				deviceMountPath)
			if err != nil {
				glog.Errorf(
					"MountVolume.MountDevice failed for volume %q (spec.Name: %q) pod %q (UID: %q) with: %v",
					volumeToMount.VolumeName,
					volumeToMount.VolumeSpec.Name(),
					volumeToMount.PodName,
					volumeToMount.Pod.UID,
					err)
				return err
			}

			glog.Infof(
				"MountVolume.MountDevice succeeded for volume %q (spec.Name: %q) pod %q (UID: %q).",
				volumeToMount.VolumeName,
				volumeToMount.VolumeSpec.Name(),
				volumeToMount.PodName,
				volumeToMount.Pod.UID)

			// Update actual state of world to reflect volume is globally mounted
			markDeviceMountedErr := actualStateOfWorld.MarkDeviceAsMounted(
				volumeToMount.VolumeName)
			if markDeviceMountedErr != nil {
				// On failure, just log and exit. The controller will retry
				glog.Errorf(
					"MountVolume.MarkDeviceAsMounted failed for volume %q (spec.Name: %q) pod %q (UID: %q) with: %v",
					volumeToMount.VolumeName,
					volumeToMount.VolumeSpec.Name(),
					volumeToMount.PodName,
					volumeToMount.Pod.UID,
					markDeviceMountedErr)
				return markDeviceMountedErr
			}
		}

		// Execute mount
		mountErr := volumeMounter.SetUp(fsGroup)
		if mountErr != nil {
			// On failure, just log and exit. The controller will retry
			glog.Errorf(
				"MountVolume.SetUp failed for volume %q (spec.Name: %q) pod %q (UID: %q) with: %v",
				volumeToMount.VolumeName,
				volumeToMount.VolumeSpec.Name(),
				volumeToMount.PodName,
				volumeToMount.Pod.UID,
				mountErr)
			return mountErr
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
			// On failure, just log and exit. The controller will retry
			glog.Errorf(
				"MountVolume.MarkVolumeAsMounted failed for volume %q (spec.Name: %q) pod %q (UID: %q) with: %v",
				volumeToMount.VolumeName,
				volumeToMount.VolumeSpec.Name(),
				volumeToMount.PodName,
				volumeToMount.Pod.UID,
				markVolMountedErr)
			return markVolMountedErr
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
			// On failure, just log and exit. The controller will retry
			glog.Errorf(
				"UnmountVolume.TearDown failed for volume %q (volume.spec.Name: %q) pod %q (UID: %q) with: %v",
				volumeToUnmount.VolumeName,
				volumeToUnmount.OuterVolumeSpecName,
				volumeToUnmount.PodName,
				volumeToUnmount.PodUID,
				unmountErr)
			return unmountErr
		}

		glog.Infof(
			"UnmountVolume.TearDown succeeded for volume %q (volume.spec.Name: %q) pod %q (UID: %q).",
			volumeToUnmount.VolumeName,
			volumeToUnmount.OuterVolumeSpecName,
			volumeToUnmount.PodName,
			volumeToUnmount.PodUID)

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
				unmountErr)
		}

		return nil
	}, nil
}

func (oe *operationExecutor) generateUnmountDeviceFunc(
	deviceToDetach AttachedVolume,
	actualStateOfWorld ActualStateOfWorldMounterUpdater) (func() error, error) {
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
			// On failure, just log and exit. The controller will retry
			glog.Errorf(
				"GetDeviceMountPath failed for volume %q (spec.Name: %q) with: %v",
				deviceToDetach.VolumeName,
				deviceToDetach.VolumeSpec.Name(),
				err)
			return err
		}

		// Execute unmount
		unmountDeviceErr := volumeDetacher.UnmountDevice(deviceMountPath)
		if unmountDeviceErr != nil {
			// On failure, just log and exit. The controller will retry
			glog.Errorf(
				"UnmountDevice failed for volume %q (spec.Name: %q) with: %v",
				deviceToDetach.VolumeName,
				deviceToDetach.VolumeSpec.Name(),
				unmountDeviceErr)
			return unmountDeviceErr
		}

		glog.Infof(
			"UnmountDevice succeeded for volume %q (spec.Name: %q).",
			deviceToDetach.VolumeName,
			deviceToDetach.VolumeSpec.Name())

		// Update actual state of world
		markDeviceUnmountedErr := actualStateOfWorld.MarkDeviceAsUnmounted(
			deviceToDetach.VolumeName)
		if markDeviceUnmountedErr != nil {
			// On failure, just log and exit. The controller will retry
			glog.Errorf(
				"MarkDeviceAsUnmounted failed for device %q (spec.Name: %q) with: %v",
				deviceToDetach.VolumeName,
				deviceToDetach.VolumeSpec.Name(),
				markDeviceUnmountedErr)
			return markDeviceUnmountedErr
		}

		return nil
	}, nil
}
