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

	"k8s.io/api/core/v1"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/util"
	"k8s.io/kubernetes/pkg/volume/util/operationexecutor"
	"k8s.io/kubernetes/pkg/volume/util/types"
)

// DesiredStateOfWorld defines a set of thread-safe operations for the kubelet
// volume manager's desired state of the world cache.
// This cache contains volumes->pods i.e. a set of all volumes that should be
// attached to this node and the pods that reference them and should mount the
// volume.
// Note: This is distinct from the DesiredStateOfWorld implemented by the
// attach/detach controller. They both keep track of different objects. This
// contains kubelet volume manager specific state.
type DesiredStateOfWorld interface {
	// AddPodToVolume adds the given pod to the given volume in the cache
	// indicating the specified pod should mount the specified volume.
	// A unique volumeName is generated from the volumeSpec and returned on
	// success.
	// If no volume plugin can support the given volumeSpec or more than one
	// plugin can support it, an error is returned.
	// If a volume with the name volumeName does not exist in the list of
	// volumes that should be attached to this node, the volume is implicitly
	// added.
	// If a pod with the same unique name already exists under the specified
	// volume, this is a no-op.
	AddPodToVolume(podName types.UniquePodName, pod *v1.Pod, volumeSpec *volume.Spec, outerVolumeSpecName string, volumeGidValue string) (v1.UniqueVolumeName, error)

	// MarkVolumesReportedInUse sets the ReportedInUse value to true for the
	// reportedVolumes. For volumes not in the reportedVolumes list, the
	// ReportedInUse value is reset to false. The default ReportedInUse value
	// for a newly created volume is false.
	// When set to true this value indicates that the volume was successfully
	// added to the VolumesInUse field in the node's status. Mount operation needs
	// to check this value before issuing the operation.
	// If a volume in the reportedVolumes list does not exist in the list of
	// volumes that should be attached to this node, it is skipped without error.
	MarkVolumesReportedInUse(reportedVolumes []v1.UniqueVolumeName)

	// DeletePodFromVolume removes the given pod from the given volume in the
	// cache indicating the specified pod no longer requires the specified
	// volume.
	// If a pod with the same unique name does not exist under the specified
	// volume, this is a no-op.
	// If a volume with the name volumeName does not exist in the list of
	// attached volumes, this is a no-op.
	// If after deleting the pod, the specified volume contains no other child
	// pods, the volume is also deleted.
	DeletePodFromVolume(podName types.UniquePodName, volumeName v1.UniqueVolumeName)

	// VolumeExists returns true if the given volume exists in the list of
	// volumes that should be attached to this node.
	// If a pod with the same unique name does not exist under the specified
	// volume, false is returned.
	VolumeExists(volumeName v1.UniqueVolumeName) bool

	// PodExistsInVolume returns true if the given pod exists in the list of
	// podsToMount for the given volume in the cache.
	// If a pod with the same unique name does not exist under the specified
	// volume, false is returned.
	// If a volume with the name volumeName does not exist in the list of
	// attached volumes, false is returned.
	PodExistsInVolume(podName types.UniquePodName, volumeName v1.UniqueVolumeName) bool

	// GetVolumesToMount generates and returns a list of volumes that should be
	// attached to this node and the pods they should be mounted to based on the
	// current desired state of the world.
	GetVolumesToMount() []VolumeToMount

	// GetPods generates and returns a map of pods in which map is indexed
	// with pod's unique name. This map can be used to determine which pod is currently
	// in desired state of world.
	GetPods() map[types.UniquePodName]bool

	// VolumeExistsWithSpecName returns true if the given volume specified with the
	// volume spec name (a.k.a., InnerVolumeSpecName) exists in the list of
	// volumes that should be attached to this node.
	// If a pod with the same name does not exist under the specified
	// volume, false is returned.
	VolumeExistsWithSpecName(podName types.UniquePodName, volumeSpecName string) bool
}

// VolumeToMount represents a volume that is attached to this node and needs to
// be mounted to PodName.
type VolumeToMount struct {
	operationexecutor.VolumeToMount
}

// NewDesiredStateOfWorld returns a new instance of DesiredStateOfWorld.
func NewDesiredStateOfWorld(volumePluginMgr *volume.VolumePluginMgr) DesiredStateOfWorld {
	return &desiredStateOfWorld{
		volumesToMount:  make(map[v1.UniqueVolumeName]volumeToMount),
		volumePluginMgr: volumePluginMgr,
	}
}

type desiredStateOfWorld struct {
	// volumesToMount is a map containing the set of volumes that should be
	// attached to this node and mounted to the pods referencing it. The key in
	// the map is the name of the volume and the value is a volume object
	// containing more information about the volume.
	volumesToMount map[v1.UniqueVolumeName]volumeToMount
	// volumePluginMgr is the volume plugin manager used to create volume
	// plugin objects.
	volumePluginMgr *volume.VolumePluginMgr

	sync.RWMutex
}

// The volume object represents a volume that should be attached to this node,
// and mounted to podsToMount.
type volumeToMount struct {
	// volumeName contains the unique identifier for this volume.
	volumeName v1.UniqueVolumeName

	// podsToMount is a map containing the set of pods that reference this
	// volume and should mount it once it is attached. The key in the map is
	// the name of the pod and the value is a pod object containing more
	// information about the pod.
	podsToMount map[types.UniquePodName]podToMount

	// pluginIsAttachable indicates that the plugin for this volume implements
	// the volume.Attacher interface
	pluginIsAttachable bool

	// pluginIsDeviceMountable indicates that the plugin for this volume implements
	// the volume.DeviceMounter interface
	pluginIsDeviceMountable bool

	// volumeGidValue contains the value of the GID annotation, if present.
	volumeGidValue string

	// reportedInUse indicates that the volume was successfully added to the
	// VolumesInUse field in the node's status.
	reportedInUse bool
}

// The pod object represents a pod that references the underlying volume and
// should mount it once it is attached.
type podToMount struct {
	// podName contains the name of this pod.
	podName types.UniquePodName

	// Pod to mount the volume to. Used to create NewMounter.
	pod *v1.Pod

	// volume spec containing the specification for this volume. Used to
	// generate the volume plugin object, and passed to plugin methods.
	// For non-PVC volumes this is the same as defined in the pod object. For
	// PVC volumes it is from the dereferenced PV object.
	volumeSpec *volume.Spec

	// outerVolumeSpecName is the volume.Spec.Name() of the volume as referenced
	// directly in the pod. If the volume was referenced through a persistent
	// volume claim, this contains the volume.Spec.Name() of the persistent
	// volume claim
	outerVolumeSpecName string
}

func (dsw *desiredStateOfWorld) AddPodToVolume(
	podName types.UniquePodName,
	pod *v1.Pod,
	volumeSpec *volume.Spec,
	outerVolumeSpecName string,
	volumeGidValue string) (v1.UniqueVolumeName, error) {
	dsw.Lock()
	defer dsw.Unlock()

	volumePlugin, err := dsw.volumePluginMgr.FindPluginBySpec(volumeSpec)
	if err != nil || volumePlugin == nil {
		return "", fmt.Errorf(
			"failed to get Plugin from volumeSpec for volume %q err=%v",
			volumeSpec.Name(),
			err)
	}

	var volumeName v1.UniqueVolumeName

	// The unique volume name used depends on whether the volume is attachable/device-mountable
	// or not.
	attachable := dsw.isAttachableVolume(volumeSpec)
	deviceMountable := dsw.isDeviceMountableVolume(volumeSpec)
	if attachable || deviceMountable {
		// For attachable/device-mountable volumes, use the unique volume name as reported by
		// the plugin.
		volumeName, err =
			util.GetUniqueVolumeNameFromSpec(volumePlugin, volumeSpec)
		if err != nil {
			return "", fmt.Errorf(
				"failed to GetUniqueVolumeNameFromSpec for volumeSpec %q using volume plugin %q err=%v",
				volumeSpec.Name(),
				volumePlugin.GetPluginName(),
				err)
		}
	} else {
		// For non-attachable and non-device-mountable volumes, generate a unique name based on the pod
		// namespace and name and the name of the volume within the pod.
		volumeName = util.GetUniqueVolumeNameFromSpecWithPod(podName, volumePlugin, volumeSpec)
	}

	if _, volumeExists := dsw.volumesToMount[volumeName]; !volumeExists {
		dsw.volumesToMount[volumeName] = volumeToMount{
			volumeName:              volumeName,
			podsToMount:             make(map[types.UniquePodName]podToMount),
			pluginIsAttachable:      attachable,
			pluginIsDeviceMountable: deviceMountable,
			volumeGidValue:          volumeGidValue,
			reportedInUse:           false,
		}
	}

	// Create new podToMount object. If it already exists, it is refreshed with
	// updated values (this is required for volumes that require remounting on
	// pod update, like Downward API volumes).
	dsw.volumesToMount[volumeName].podsToMount[podName] = podToMount{
		podName:             podName,
		pod:                 pod,
		volumeSpec:          volumeSpec,
		outerVolumeSpecName: outerVolumeSpecName,
	}
	return volumeName, nil
}

func (dsw *desiredStateOfWorld) MarkVolumesReportedInUse(
	reportedVolumes []v1.UniqueVolumeName) {
	dsw.Lock()
	defer dsw.Unlock()

	reportedVolumesMap := make(
		map[v1.UniqueVolumeName]bool, len(reportedVolumes) /* capacity */)

	for _, reportedVolume := range reportedVolumes {
		reportedVolumesMap[reportedVolume] = true
	}

	for volumeName, volumeObj := range dsw.volumesToMount {
		_, volumeReported := reportedVolumesMap[volumeName]
		volumeObj.reportedInUse = volumeReported
		dsw.volumesToMount[volumeName] = volumeObj
	}
}

func (dsw *desiredStateOfWorld) DeletePodFromVolume(
	podName types.UniquePodName, volumeName v1.UniqueVolumeName) {
	dsw.Lock()
	defer dsw.Unlock()

	volumeObj, volumeExists := dsw.volumesToMount[volumeName]
	if !volumeExists {
		return
	}

	if _, podExists := volumeObj.podsToMount[podName]; !podExists {
		return
	}

	// Delete pod if it exists
	delete(dsw.volumesToMount[volumeName].podsToMount, podName)

	if len(dsw.volumesToMount[volumeName].podsToMount) == 0 {
		// Delete volume if no child pods left
		delete(dsw.volumesToMount, volumeName)
	}
}

func (dsw *desiredStateOfWorld) VolumeExists(
	volumeName v1.UniqueVolumeName) bool {
	dsw.RLock()
	defer dsw.RUnlock()

	_, volumeExists := dsw.volumesToMount[volumeName]
	return volumeExists
}

func (dsw *desiredStateOfWorld) PodExistsInVolume(
	podName types.UniquePodName, volumeName v1.UniqueVolumeName) bool {
	dsw.RLock()
	defer dsw.RUnlock()

	volumeObj, volumeExists := dsw.volumesToMount[volumeName]
	if !volumeExists {
		return false
	}

	_, podExists := volumeObj.podsToMount[podName]
	return podExists
}

func (dsw *desiredStateOfWorld) VolumeExistsWithSpecName(podName types.UniquePodName, volumeSpecName string) bool {
	dsw.RLock()
	defer dsw.RUnlock()
	for _, volumeObj := range dsw.volumesToMount {
		for name, podObj := range volumeObj.podsToMount {
			if podName == name && podObj.volumeSpec.Name() == volumeSpecName {
				return true
			}
		}
	}
	return false
}

func (dsw *desiredStateOfWorld) GetPods() map[types.UniquePodName]bool {
	dsw.RLock()
	defer dsw.RUnlock()

	podList := make(map[types.UniquePodName]bool)
	for _, volumeObj := range dsw.volumesToMount {
		for podName := range volumeObj.podsToMount {
			if !podList[podName] {
				podList[podName] = true
			}
		}
	}
	return podList
}

func (dsw *desiredStateOfWorld) GetVolumesToMount() []VolumeToMount {
	dsw.RLock()
	defer dsw.RUnlock()

	volumesToMount := make([]VolumeToMount, 0 /* len */, len(dsw.volumesToMount) /* cap */)
	for volumeName, volumeObj := range dsw.volumesToMount {
		for podName, podObj := range volumeObj.podsToMount {
			volumesToMount = append(
				volumesToMount,
				VolumeToMount{
					VolumeToMount: operationexecutor.VolumeToMount{
						VolumeName:              volumeName,
						PodName:                 podName,
						Pod:                     podObj.pod,
						VolumeSpec:              podObj.volumeSpec,
						PluginIsAttachable:      volumeObj.pluginIsAttachable,
						PluginIsDeviceMountable: volumeObj.pluginIsDeviceMountable,
						OuterVolumeSpecName:     podObj.outerVolumeSpecName,
						VolumeGidValue:          volumeObj.volumeGidValue,
						ReportedInUse:           volumeObj.reportedInUse}})
		}
	}
	return volumesToMount
}

func (dsw *desiredStateOfWorld) isAttachableVolume(volumeSpec *volume.Spec) bool {
	attachableVolumePlugin, _ :=
		dsw.volumePluginMgr.FindAttachablePluginBySpec(volumeSpec)
	if attachableVolumePlugin != nil {
		volumeAttacher, err := attachableVolumePlugin.NewAttacher()
		if err == nil && volumeAttacher != nil {
			return true
		}
	}

	return false
}

func (dsw *desiredStateOfWorld) isDeviceMountableVolume(volumeSpec *volume.Spec) bool {
	deviceMountableVolumePlugin, _ := dsw.volumePluginMgr.FindDeviceMountablePluginBySpec(volumeSpec)
	if deviceMountableVolumePlugin != nil {
		volumeDeviceMounter, err := deviceMountableVolumePlugin.NewDeviceMounter()
		if err == nil && volumeDeviceMounter != nil {
			return true
		}
	}

	return false
}
