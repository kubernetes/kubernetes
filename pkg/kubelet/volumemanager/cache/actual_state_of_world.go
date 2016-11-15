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
Package cache implements data structures used by the kubelet volume manager to
keep track of attached volumes and the pods that mounted them.
*/
package cache

import (
	"fmt"
	"sync"

	"github.com/golang/glog"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/util/operationexecutor"
	volumetypes "k8s.io/kubernetes/pkg/volume/util/types"
	"k8s.io/kubernetes/pkg/volume/util/volumehelper"
)

// ActualStateOfWorld defines a set of thread-safe operations for the kubelet
// volume manager's actual state of the world cache.
// This cache contains volumes->pods i.e. a set of all volumes attached to this
// node and the pods that the manager believes have successfully mounted the
// volume.
// Note: This is distinct from the ActualStateOfWorld implemented by the
// attach/detach controller. They both keep track of different objects. This
// contains kubelet volume manager specific state.
type ActualStateOfWorld interface {
	// ActualStateOfWorld must implement the methods required to allow
	// operationexecutor to interact with it.
	operationexecutor.ActualStateOfWorldMounterUpdater

	// ActualStateOfWorld must implement the methods required to allow
	// operationexecutor to interact with it.
	operationexecutor.ActualStateOfWorldAttacherUpdater

	// AddPodToVolume adds the given pod to the given volume in the cache
	// indicating the specified volume has been successfully mounted to the
	// specified pod.
	// If a pod with the same unique name already exists under the specified
	// volume, this is a no-op.
	// If a volume with the name volumeName does not exist in the list of
	// attached volumes, an error is returned.
	AddPodToVolume(podName volumetypes.UniquePodName, podUID types.UID, volumeName api.UniqueVolumeName, mounter volume.Mounter, outerVolumeSpecName string, volumeGidValue string) error

	// MarkRemountRequired marks each volume that is successfully attached and
	// mounted for the specified pod as requiring remount (if the plugin for the
	// volume indicates it requires remounting on pod updates). Atomically
	// updating volumes depend on this to update the contents of the volume on
	// pod update.
	MarkRemountRequired(podName volumetypes.UniquePodName)

	// SetVolumeGloballyMounted sets the GloballyMounted value for the given
	// volume. When set to true this value indicates that the volume is mounted
	// to the underlying device at a global mount point. This global mount point
	// must unmounted prior to detach.
	// If a volume with the name volumeName does not exist in the list of
	// attached volumes, an error is returned.
	SetVolumeGloballyMounted(volumeName api.UniqueVolumeName, globallyMounted bool) error

	// DeletePodFromVolume removes the given pod from the given volume in the
	// cache indicating the volume has been successfully unmounted from the pod.
	// If a pod with the same unique name does not exist under the specified
	// volume, this is a no-op.
	// If a volume with the name volumeName does not exist in the list of
	// attached volumes, an error is returned.
	DeletePodFromVolume(podName volumetypes.UniquePodName, volumeName api.UniqueVolumeName) error

	// DeleteVolume removes the given volume from the list of attached volumes
	// in the cache indicating the volume has been successfully detached from
	// this node.
	// If a volume with the name volumeName does not exist in the list of
	// attached volumes, this is a no-op.
	// If a volume with the name volumeName exists and its list of mountedPods
	// is not empty, an error is returned.
	DeleteVolume(volumeName api.UniqueVolumeName) error

	// PodExistsInVolume returns true if the given pod exists in the list of
	// mountedPods for the given volume in the cache, indicating that the volume
	// is attached to this node and the pod has successfully mounted it.
	// If a pod with the same unique name does not exist under the specified
	// volume, false is returned.
	// If a volume with the name volumeName does not exist in the list of
	// attached volumes, a volumeNotAttachedError is returned indicating the
	// given volume is not yet attached.
	// If the given volumeName/podName combo exists but the value of
	// remountRequired is true, a remountRequiredError is returned indicating
	// the given volume has been successfully mounted to this pod but should be
	// remounted to reflect changes in the referencing pod. Atomically updating
	// volumes, depend on this to update the contents of the volume.
	// All volume mounting calls should be idempotent so a second mount call for
	// volumes that do not need to update contents should not fail.
	PodExistsInVolume(podName volumetypes.UniquePodName, volumeName api.UniqueVolumeName) (bool, string, error)

	// VolumeExists returns true if the given volume exists in the list of
	// attached volumes in the cache, indicating the volume is attached to this
	// node.
	VolumeExists(volumeName api.UniqueVolumeName) bool

	// GetMountedVolumes generates and returns a list of volumes and the pods
	// they are successfully attached and mounted for based on the current
	// actual state of the world.
	GetMountedVolumes() []MountedVolume

	// GetMountedVolumesForPod generates and returns a list of volumes that are
	// successfully attached and mounted for the specified pod based on the
	// current actual state of the world.
	GetMountedVolumesForPod(podName volumetypes.UniquePodName) []MountedVolume

	// GetGloballyMountedVolumes generates and returns a list of all attached
	// volumes that are globally mounted. This list can be used to determine
	// which volumes should be reported as "in use" in the node's VolumesInUse
	// status field. Globally mounted here refers to the shared plugin mount
	// point for the attachable volume from which the pod specific mount points
	// are created (via bind mount).
	GetGloballyMountedVolumes() []AttachedVolume

	// GetUnmountedVolumes generates and returns a list of attached volumes that
	// have no mountedPods. This list can be used to determine which volumes are
	// no longer referenced and may be globally unmounted and detached.
	GetUnmountedVolumes() []AttachedVolume

	// GetPods generates and returns a map of pods in which map is indexed
	// with pod's unique name. This map can be used to determine which pod is currently
	// in actual state of world.
	GetPods() map[volumetypes.UniquePodName]bool
}

// MountedVolume represents a volume that has successfully been mounted to a pod.
type MountedVolume struct {
	operationexecutor.MountedVolume
}

// AttachedVolume represents a volume that is attached to a node.
type AttachedVolume struct {
	operationexecutor.AttachedVolume

	// GloballyMounted indicates that the volume is mounted to the underlying
	// device at a global mount point. This global mount point must unmounted
	// prior to detach.
	GloballyMounted bool
}

// NewActualStateOfWorld returns a new instance of ActualStateOfWorld.
func NewActualStateOfWorld(
	nodeName types.NodeName,
	volumePluginMgr *volume.VolumePluginMgr) ActualStateOfWorld {
	return &actualStateOfWorld{
		nodeName:        nodeName,
		attachedVolumes: make(map[api.UniqueVolumeName]attachedVolume),
		volumePluginMgr: volumePluginMgr,
	}
}

// IsVolumeNotAttachedError returns true if the specified error is a
// volumeNotAttachedError.
func IsVolumeNotAttachedError(err error) bool {
	_, ok := err.(volumeNotAttachedError)
	return ok
}

// IsRemountRequiredError returns true if the specified error is a
// remountRequiredError.
func IsRemountRequiredError(err error) bool {
	_, ok := err.(remountRequiredError)
	return ok
}

type actualStateOfWorld struct {
	// nodeName is the name of this node. This value is passed to Attach/Detach
	nodeName types.NodeName

	// attachedVolumes is a map containing the set of volumes the kubelet volume
	// manager believes to be successfully attached to this node. Volume types
	// that do not implement an attacher interface are assumed to be in this
	// state by default.
	// The key in this map is the name of the volume and the value is an object
	// containing more information about the attached volume.
	attachedVolumes map[api.UniqueVolumeName]attachedVolume

	// volumePluginMgr is the volume plugin manager used to create volume
	// plugin objects.
	volumePluginMgr *volume.VolumePluginMgr
	sync.RWMutex
}

// attachedVolume represents a volume the kubelet volume manager believes to be
// successfully attached to a node it is managing. Volume types that do not
// implement an attacher are assumed to be in this state.
type attachedVolume struct {
	// volumeName contains the unique identifier for this volume.
	volumeName api.UniqueVolumeName

	// mountedPods is a map containing the set of pods that this volume has been
	// successfully mounted to. The key in this map is the name of the pod and
	// the value is a mountedPod object containing more information about the
	// pod.
	mountedPods map[volumetypes.UniquePodName]mountedPod

	// spec is the volume spec containing the specification for this volume.
	// Used to generate the volume plugin object, and passed to plugin methods.
	// In particular, the Unmount method uses spec.Name() as the volumeSpecName
	// in the mount path:
	// /var/lib/kubelet/pods/{podUID}/volumes/{escapeQualifiedPluginName}/{volumeSpecName}/
	spec *volume.Spec

	// pluginName is the Unescaped Qualified name of the volume plugin used to
	// attach and mount this volume. It is stored separately in case the full
	// volume spec (everything except the name) can not be reconstructed for a
	// volume that should be unmounted (which would be the case for a mount path
	// read from disk without a full volume spec).
	pluginName string

	// pluginIsAttachable indicates the volume plugin used to attach and mount
	// this volume implements the volume.Attacher interface
	pluginIsAttachable bool

	// globallyMounted indicates that the volume is mounted to the underlying
	// device at a global mount point. This global mount point must be unmounted
	// prior to detach.
	globallyMounted bool

	// devicePath contains the path on the node where the volume is attached for
	// attachable volumes
	devicePath string
}

// The mountedPod object represents a pod for which the kubelet volume manager
// believes the underlying volume has been successfully been mounted.
type mountedPod struct {
	// the name of the pod
	podName volumetypes.UniquePodName

	// the UID of the pod
	podUID types.UID

	// mounter used to mount
	mounter volume.Mounter

	// outerVolumeSpecName is the volume.Spec.Name() of the volume as referenced
	// directly in the pod. If the volume was referenced through a persistent
	// volume claim, this contains the volume.Spec.Name() of the persistent
	// volume claim
	outerVolumeSpecName string

	// remountRequired indicates the underlying volume has been successfully
	// mounted to this pod but it should be remounted to reflect changes in the
	// referencing pod.
	// Atomically updating volumes depend on this to update the contents of the
	// volume. All volume mounting calls should be idempotent so a second mount
	// call for volumes that do not need to update contents should not fail.
	remountRequired bool

	// volumeGidValue contains the value of the GID annotation, if present.
	volumeGidValue string
}

func (asw *actualStateOfWorld) MarkVolumeAsAttached(
	volumeName api.UniqueVolumeName, volumeSpec *volume.Spec, _ types.NodeName, devicePath string) error {
	return asw.addVolume(volumeName, volumeSpec, devicePath)
}

func (asw *actualStateOfWorld) MarkVolumeAsDetached(
	volumeName api.UniqueVolumeName, nodeName types.NodeName) {
	asw.DeleteVolume(volumeName)
}

func (asw *actualStateOfWorld) MarkVolumeAsMounted(
	podName volumetypes.UniquePodName,
	podUID types.UID,
	volumeName api.UniqueVolumeName,
	mounter volume.Mounter,
	outerVolumeSpecName string,
	volumeGidValue string) error {
	return asw.AddPodToVolume(
		podName,
		podUID,
		volumeName,
		mounter,
		outerVolumeSpecName,
		volumeGidValue)
}

func (asw *actualStateOfWorld) AddVolumeToReportAsAttached(volumeName api.UniqueVolumeName, nodeName types.NodeName) {
	// no operation for kubelet side
}

func (asw *actualStateOfWorld) RemoveVolumeFromReportAsAttached(volumeName api.UniqueVolumeName, nodeName types.NodeName) error {
	// no operation for kubelet side
	return nil
}

func (asw *actualStateOfWorld) MarkVolumeAsUnmounted(
	podName volumetypes.UniquePodName, volumeName api.UniqueVolumeName) error {
	return asw.DeletePodFromVolume(podName, volumeName)
}

func (asw *actualStateOfWorld) MarkDeviceAsMounted(
	volumeName api.UniqueVolumeName) error {
	return asw.SetVolumeGloballyMounted(volumeName, true /* globallyMounted */)
}

func (asw *actualStateOfWorld) MarkDeviceAsUnmounted(
	volumeName api.UniqueVolumeName) error {
	return asw.SetVolumeGloballyMounted(volumeName, false /* globallyMounted */)
}

// addVolume adds the given volume to the cache indicating the specified
// volume is attached to this node. If no volume name is supplied, a unique
// volume name is generated from the volumeSpec and returned on success. If a
// volume with the same generated name already exists, this is a noop. If no
// volume plugin can support the given volumeSpec or more than one plugin can
// support it, an error is returned.
func (asw *actualStateOfWorld) addVolume(
	volumeName api.UniqueVolumeName, volumeSpec *volume.Spec, devicePath string) error {
	asw.Lock()
	defer asw.Unlock()

	volumePlugin, err := asw.volumePluginMgr.FindPluginBySpec(volumeSpec)
	if err != nil || volumePlugin == nil {
		return fmt.Errorf(
			"failed to get Plugin from volumeSpec for volume %q err=%v",
			volumeSpec.Name(),
			err)
	}

	if len(volumeName) == 0 {
		volumeName, err = volumehelper.GetUniqueVolumeNameFromSpec(volumePlugin, volumeSpec)
		if err != nil {
			return fmt.Errorf(
				"failed to GetUniqueVolumeNameFromSpec for volumeSpec %q using volume plugin %q err=%v",
				volumeSpec.Name(),
				volumePlugin.GetPluginName(),
				err)
		}
	}

	pluginIsAttachable := false
	if _, ok := volumePlugin.(volume.AttachableVolumePlugin); ok {
		pluginIsAttachable = true
	}

	volumeObj, volumeExists := asw.attachedVolumes[volumeName]
	if !volumeExists {
		volumeObj = attachedVolume{
			volumeName:         volumeName,
			spec:               volumeSpec,
			mountedPods:        make(map[volumetypes.UniquePodName]mountedPod),
			pluginName:         volumePlugin.GetPluginName(),
			pluginIsAttachable: pluginIsAttachable,
			globallyMounted:    false,
			devicePath:         devicePath,
		}
	} else {
		// If volume object already exists, update the fields such as device path
		volumeObj.devicePath = devicePath
		glog.V(2).Infof("Volume %q is already added to attachedVolume list, update device path %q",
			volumeName,
			devicePath)
	}
	asw.attachedVolumes[volumeName] = volumeObj

	return nil
}

func (asw *actualStateOfWorld) AddPodToVolume(
	podName volumetypes.UniquePodName,
	podUID types.UID,
	volumeName api.UniqueVolumeName,
	mounter volume.Mounter,
	outerVolumeSpecName string,
	volumeGidValue string) error {
	asw.Lock()
	defer asw.Unlock()

	volumeObj, volumeExists := asw.attachedVolumes[volumeName]
	if !volumeExists {
		return fmt.Errorf(
			"no volume with the name %q exists in the list of attached volumes",
			volumeName)
	}

	podObj, podExists := volumeObj.mountedPods[podName]
	if !podExists {
		podObj = mountedPod{
			podName:             podName,
			podUID:              podUID,
			mounter:             mounter,
			outerVolumeSpecName: outerVolumeSpecName,
			volumeGidValue:      volumeGidValue,
		}
	}

	// If pod exists, reset remountRequired value
	podObj.remountRequired = false
	asw.attachedVolumes[volumeName].mountedPods[podName] = podObj

	return nil
}

func (asw *actualStateOfWorld) MarkRemountRequired(
	podName volumetypes.UniquePodName) {
	asw.Lock()
	defer asw.Unlock()
	for volumeName, volumeObj := range asw.attachedVolumes {
		for mountedPodName, podObj := range volumeObj.mountedPods {
			if mountedPodName != podName {
				continue
			}

			volumePlugin, err :=
				asw.volumePluginMgr.FindPluginBySpec(volumeObj.spec)
			if err != nil || volumePlugin == nil {
				// Log and continue processing
				glog.Errorf(
					"MarkRemountRequired failed to FindPluginBySpec for pod %q (podUid %q) volume: %q (volSpecName: %q)",
					podObj.podName,
					podObj.podUID,
					volumeObj.volumeName,
					volumeObj.spec.Name())
				continue
			}

			if volumePlugin.RequiresRemount() {
				podObj.remountRequired = true
				asw.attachedVolumes[volumeName].mountedPods[podName] = podObj
			}
		}
	}
}

func (asw *actualStateOfWorld) SetVolumeGloballyMounted(
	volumeName api.UniqueVolumeName, globallyMounted bool) error {
	asw.Lock()
	defer asw.Unlock()

	volumeObj, volumeExists := asw.attachedVolumes[volumeName]
	if !volumeExists {
		return fmt.Errorf(
			"no volume with the name %q exists in the list of attached volumes",
			volumeName)
	}

	volumeObj.globallyMounted = globallyMounted
	asw.attachedVolumes[volumeName] = volumeObj
	return nil
}

func (asw *actualStateOfWorld) DeletePodFromVolume(
	podName volumetypes.UniquePodName, volumeName api.UniqueVolumeName) error {
	asw.Lock()
	defer asw.Unlock()

	volumeObj, volumeExists := asw.attachedVolumes[volumeName]
	if !volumeExists {
		return fmt.Errorf(
			"no volume with the name %q exists in the list of attached volumes",
			volumeName)
	}

	_, podExists := volumeObj.mountedPods[podName]
	if podExists {
		delete(asw.attachedVolumes[volumeName].mountedPods, podName)
	}

	return nil
}

func (asw *actualStateOfWorld) DeleteVolume(volumeName api.UniqueVolumeName) error {
	asw.Lock()
	defer asw.Unlock()

	volumeObj, volumeExists := asw.attachedVolumes[volumeName]
	if !volumeExists {
		return nil
	}

	if len(volumeObj.mountedPods) != 0 {
		return fmt.Errorf(
			"failed to DeleteVolume %q, it still has %v mountedPods",
			volumeName,
			len(volumeObj.mountedPods))
	}

	delete(asw.attachedVolumes, volumeName)
	return nil
}

func (asw *actualStateOfWorld) PodExistsInVolume(
	podName volumetypes.UniquePodName,
	volumeName api.UniqueVolumeName) (bool, string, error) {
	asw.RLock()
	defer asw.RUnlock()

	volumeObj, volumeExists := asw.attachedVolumes[volumeName]
	if !volumeExists {
		return false, "", newVolumeNotAttachedError(volumeName)
	}

	podObj, podExists := volumeObj.mountedPods[podName]
	if podExists && podObj.remountRequired {
		return true, volumeObj.devicePath, newRemountRequiredError(volumeObj.volumeName, podObj.podName)
	}

	return podExists, volumeObj.devicePath, nil
}

func (asw *actualStateOfWorld) VolumeExists(
	volumeName api.UniqueVolumeName) bool {
	asw.RLock()
	defer asw.RUnlock()

	_, volumeExists := asw.attachedVolumes[volumeName]
	return volumeExists
}

func (asw *actualStateOfWorld) GetMountedVolumes() []MountedVolume {
	asw.RLock()
	defer asw.RUnlock()
	mountedVolume := make([]MountedVolume, 0 /* len */, len(asw.attachedVolumes) /* cap */)
	for _, volumeObj := range asw.attachedVolumes {
		for _, podObj := range volumeObj.mountedPods {
			mountedVolume = append(
				mountedVolume,
				getMountedVolume(&podObj, &volumeObj))
		}
	}

	return mountedVolume
}

func (asw *actualStateOfWorld) GetMountedVolumesForPod(
	podName volumetypes.UniquePodName) []MountedVolume {
	asw.RLock()
	defer asw.RUnlock()
	mountedVolume := make([]MountedVolume, 0 /* len */, len(asw.attachedVolumes) /* cap */)
	for _, volumeObj := range asw.attachedVolumes {
		for mountedPodName, podObj := range volumeObj.mountedPods {
			if mountedPodName == podName {
				mountedVolume = append(
					mountedVolume,
					getMountedVolume(&podObj, &volumeObj))
			}
		}
	}

	return mountedVolume
}

func (asw *actualStateOfWorld) GetGloballyMountedVolumes() []AttachedVolume {
	asw.RLock()
	defer asw.RUnlock()
	globallyMountedVolumes := make(
		[]AttachedVolume, 0 /* len */, len(asw.attachedVolumes) /* cap */)
	for _, volumeObj := range asw.attachedVolumes {
		if volumeObj.globallyMounted {
			globallyMountedVolumes = append(
				globallyMountedVolumes,
				asw.newAttachedVolume(&volumeObj))
		}
	}

	return globallyMountedVolumes
}

func (asw *actualStateOfWorld) GetUnmountedVolumes() []AttachedVolume {
	asw.RLock()
	defer asw.RUnlock()
	unmountedVolumes := make([]AttachedVolume, 0 /* len */, len(asw.attachedVolumes) /* cap */)
	for _, volumeObj := range asw.attachedVolumes {
		if len(volumeObj.mountedPods) == 0 {
			unmountedVolumes = append(
				unmountedVolumes,
				asw.newAttachedVolume(&volumeObj))
		}
	}

	return unmountedVolumes
}

func (asw *actualStateOfWorld) GetPods() map[volumetypes.UniquePodName]bool {
	asw.RLock()
	defer asw.RUnlock()

	podList := make(map[volumetypes.UniquePodName]bool)
	for _, volumeObj := range asw.attachedVolumes {
		for podName := range volumeObj.mountedPods {
			if !podList[podName] {
				podList[podName] = true
			}
		}
	}
	return podList
}

func (asw *actualStateOfWorld) newAttachedVolume(
	attachedVolume *attachedVolume) AttachedVolume {
	return AttachedVolume{
		AttachedVolume: operationexecutor.AttachedVolume{
			VolumeName:         attachedVolume.volumeName,
			VolumeSpec:         attachedVolume.spec,
			NodeName:           asw.nodeName,
			PluginIsAttachable: attachedVolume.pluginIsAttachable,
			DevicePath:         attachedVolume.devicePath},
		GloballyMounted: attachedVolume.globallyMounted}
}

// Compile-time check to ensure volumeNotAttachedError implements the error interface
var _ error = volumeNotAttachedError{}

// volumeNotAttachedError is an error returned when PodExistsInVolume() fails to
// find specified volume in the list of attached volumes.
type volumeNotAttachedError struct {
	volumeName api.UniqueVolumeName
}

func (err volumeNotAttachedError) Error() string {
	return fmt.Sprintf(
		"volumeName %q does not exist in the list of attached volumes",
		err.volumeName)
}

func newVolumeNotAttachedError(volumeName api.UniqueVolumeName) error {
	return volumeNotAttachedError{
		volumeName: volumeName,
	}
}

// Compile-time check to ensure remountRequiredError implements the error interface
var _ error = remountRequiredError{}

// remountRequiredError is an error returned when PodExistsInVolume() found
// volume/pod attached/mounted but remountRequired was true, indicating the
// given volume should be remounted to the pod to reflect changes in the
// referencing pod.
type remountRequiredError struct {
	volumeName api.UniqueVolumeName
	podName    volumetypes.UniquePodName
}

func (err remountRequiredError) Error() string {
	return fmt.Sprintf(
		"volumeName %q is mounted to %q but should be remounted",
		err.volumeName, err.podName)
}

func newRemountRequiredError(
	volumeName api.UniqueVolumeName, podName volumetypes.UniquePodName) error {
	return remountRequiredError{
		volumeName: volumeName,
		podName:    podName,
	}
}

// getMountedVolume constructs and returns a MountedVolume object from the given
// mountedPod and attachedVolume objects.
func getMountedVolume(
	mountedPod *mountedPod, attachedVolume *attachedVolume) MountedVolume {
	return MountedVolume{
		MountedVolume: operationexecutor.MountedVolume{
			PodName:             mountedPod.podName,
			VolumeName:          attachedVolume.volumeName,
			InnerVolumeSpecName: attachedVolume.spec.Name(),
			OuterVolumeSpecName: mountedPod.outerVolumeSpecName,
			PluginName:          attachedVolume.pluginName,
			PodUID:              mountedPod.podUID,
			Mounter:             mountedPod.mounter,
			VolumeGidValue:      mountedPod.volumeGidValue}}
}
